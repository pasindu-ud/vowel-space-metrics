import math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import linalg
from scipy.spatial import ConvexHull

from src.constants import *


class Coordinate:
    def __init__(self, name: str, f1: float, f2: float) -> None:
        # store vowel name and map f1/f2 into x,y as you requested:
        self.name: str = name
        self.x: float = f2  # x-coordinate = F2
        self.y: float = f1  # y-coordinate = F1


def hull_polygon(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ordered polygon coordinates for the convex hull of `points`.
    If < 3 points, returns the points (no hull).
    Input: points shape (N,2)
    Returns: xs, ys (closed polygon: last == first)
    """
    if len(points) < 3:
        # not enough points to form a polygon; return them directly
        xs = points[:, 0]
        ys = points[:, 1]
        return np.concatenate([xs, xs[:1]]), np.concatenate([ys, ys[:1]])
    hull = ConvexHull(points)
    poly = points[hull.vertices]  # vertices are in CCW order
    xs = np.concatenate([poly[:, 0], poly[0:1, 0]])
    ys = np.concatenate([poly[:, 1], poly[0:1, 1]])
    return xs, ys


def compute_procrustes_normalised_disparity(synth: np.ndarray,
                                            gt: np.ndarray,
                                            do_scaling: bool = True) -> Dict[str, Any]:
    """
    Align synthesized points `synth` to ground-truth `gt` using orthogonal Procrustes.
    Arguments:
      - synth: (N,2) array of [x=F2, y=F1] for synthesized centroids (in same vowel order as gt)
      - gt:    (N,2) array of ground-truth centroids
      - do_scaling: whether to allow uniform scaling (True for 'full Procrustes')

    Returns a dict containing:
      - X_transformed: transformed synth points aligned into GT space (same units as gt)
      - R: 2x2 rotation/reflection matrix found
      - scale: effective scalar applied
      - translation: vector added after rotation+scale to put into GT coordinates
      - ssq: sum of squared residuals after alignment
      - normalized_disparity: ssq divided by total GT variance (dimensionless proportion)
      - residuals: (gt - X_transformed)
      - rotation_deg: rotation angle in degrees (for interpretation)
    """

    # Step 0: basic checks
    if synth.shape != gt.shape:
        raise ValueError("synth and gt must have the same shape (same vowels in same order).")

    # Step 1: compute centroids (means) and center both point sets
    # Centering removes translation differences (shifts in x/y).
    mu_synth = synth.mean(axis=0)  # [mean_x_synth, mean_y_synth]
    mu_gt = gt.mean(axis=0)  # [mean_x_gt, mean_y_gt]
    Xc = synth - mu_synth  # centered synth
    Yc = gt - mu_gt  # centered gt

    # Step 2: compute Frobenius (matrix) norms of centered matrices.
    # The Frobenius norm sqrt(sum(entries^2)) measures overall size/scale of the point cloud.
    # We'll use these norms to normalize the shapes for stable rotation solving.
    norm_Xc = np.linalg.norm(Xc)  # sqrt of sum of squared centered coords of synth
    norm_Yc = np.linalg.norm(Yc)  # sqrt of sum of squared centered coords of gt

    if norm_Xc == 0 or norm_Yc == 0:
        raise ValueError("Zero variance found in one of the point sets; can't align.")

    # Step 3: normalize to unit size for numerical stability when finding rotation.
    # After this both Xn and Yn have Frobenius norm == 1.
    Xn = Xc / norm_Xc
    Yn = Yc / norm_Yc

    # Step 4: find best orthogonal matrix R that rotates Xn to Yn.
    # orthogonal_procrustes(A, B) returns matrix R and a scaling factor s
    # such that A @ R * s ≈ B in least-squares sense. We will combine that s with
    # the original norms to obtain the final effective scale.
    R, scale_factor = linalg.orthogonal_procrustes(Xn, Yn)
    # `R` is orthogonal (R.T @ R ≈ I). `scale_factor` is a scalar returned by SciPy.

    # Step 5: determine effective scale to map original (uncentered) synth -> gt.
    # If do_scaling is True, allow uniform scaling. Otherwise preserve scale ratio based on norms.
    if do_scaling:
        # scale_factor is found for unit-normalized Xn -> Yn; to map original Xc -> Yc:
        # effective_scale = scale_factor * (norm_Yc / norm_Xc)
        effective_scale = float(scale_factor) * (norm_Yc / norm_Xc)
    else:
        # If we forbid scaling, we still need a small consistent mapping from X -> Y:
        effective_scale = 1.0 * (norm_Yc / norm_Xc)

    # Step 6: apply transform to original centered synth Xc:
    # X_transformed_centered = effective_scale * (Xc @ R)
    X_transformed_centered = effective_scale * (Xc @ R)

    # Step 7: translate the transformed centered points to GT coordinates by adding GT centroid:
    # X_transformed = X_transformed_centered + mu_gt
    X_transformed = X_transformed_centered + mu_gt

    # Step 8: compute residuals and summary statistics
    residuals = gt - X_transformed  # vector differences per point (gt - transformed_synth)
    ssq = float(np.sum(residuals ** 2))  # sum of squared residuals (Procrustes ssq)
    denom = float(np.sum((gt - mu_gt) ** 2))  # total GT variance (sum squared deviations from mean)
    normalized_disparity = ssq / denom if denom != 0 else np.nan

    # Rotation angle (for interpretation). For a 2x2 rotation matrix R, an angle theta satisfying:
    # R ≈ [[cos theta, -sin theta],
    #      [sin theta,  cos theta]]
    # We can estimate theta from R[0,0] and R[1,0] (atan2).
    rotation_rad = math.atan2(R[1, 0], R[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    rotation_deg = -rotation_deg  # Not in the original code. Did this by me to reflect what is visualised

    # Step 9: compute the translation vector (explicit)
    translation_vec = mu_gt - (effective_scale * (mu_synth @ R))

    return {
        "X_transformed": X_transformed,
        "R": R,
        "scale": effective_scale,
        "translation": translation_vec,
        "ssq": ssq,
        "normalized_disparity": normalized_disparity,
        "residuals": residuals,
        "rotation_deg": rotation_deg
    }


def create_procrustes_animation(formant_df: pd.DataFrame,
                                corner_vowels: List[str],
                                step_prefixes,
                                accent: str,
                                allow_scaling: bool = True,
                                output_html: str = "procrustes_alignment.html"):
    """
    Build an animated Plotly figure with one frame per synthesis step.
    For each frame we:
      - align the synth corner-vowel polygon to the ground-truth polygon using Procrustes,
      - plot the GT polygon and the transformed synth polygon,
      - annotate with normalized disparity, scale and rotation.
    Inputs:
      - formant_df: dataframe with columns 'vowel', 'truth_f1','truth_f2', and step columns like '0_f1','0_f2','1000_f1','1000_f2',...
      - corner_vowels: list of vowel labels that appear in formant_df['vowel'] (order matters for correspondence)
    Returns:
      - fig (Plotly Figure), metrics (dict keyed by step)
    """

    # --- basic DataFrame setup ---
    df = formant_df.copy()
    if 'vowel' not in df.columns:
        raise ValueError("formant_df must contain a 'vowel' column.")
    df = df.set_index('vowel')

    # --- check corner vowels present ---
    missing = [v for v in corner_vowels if v not in df.index]
    if missing:
        raise ValueError(f"Corner vowels missing from DataFrame: {missing}")

    procrustes_df: pd.DataFrame = pd.DataFrame(columns=["step", "normalised_disparity", "scale", "rotation"])
    procrustes_df.loc[len(procrustes_df)] = {"step": "truth", "normalised_disparity": np.nan, "scale": np.nan,
                                             "rotation": None}

    # --- ground-truth coordinates for corner vowels (N,2) where x=F2, y=F1 ---
    gt_coords = np.vstack([[float(df.loc[v, 'truth_f2']), float(df.loc[v, 'truth_f1'])] for v in corner_vowels])

    fig = go.Figure()
    for _ in range(5):
        fig.add_trace(trace=go.Scatter())

    # prepare storage
    frames = []

    gt_hx, gt_hy = hull_polygon(gt_coords)
    traces = [
        go.Scatter(x=gt_coords[:, 0], y=gt_coords[:, 1], mode="markers+text", marker=dict(size=7, color="#D55E00"),
                   text=corner_vowels, textposition="top center", textfont=dict(color="#D55E00"),
                   hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>", name="Vowels",
                   showlegend=False),
        go.Scatter(x=gt_hx, y=gt_hy, mode='lines', line=dict(width=2, color="#D55E00"), fill="toself",
                   fillcolor="rgba(213, 94, 0, 0.15)", hoverinfo="skip", name="Ground Truth Vowel Space Shape"),
        go.Scatter(visible=False),
        go.Scatter(visible=False),
        go.Scatter(visible=False)
    ]
    # frames.append(go.Frame(data=traces, name="Ground Truth"))

    metrics: Dict[str, Dict[str, float]] = {}

    # iterate steps and create a frame for each valid step
    for step in step_prefixes:
        f1_col = f"{step}_f1"
        f2_col = f"{step}_f2"
        if f1_col not in df.columns or f2_col not in df.columns:
            # skip incomplete steps
            continue

        # synth coords for this step (N,2)
        synth_coords = np.vstack([[float(df.loc[v, f2_col]), float(df.loc[v, f1_col])] for v in corner_vowels])

        # Run Procrustes explicit routine (center -> scale (optional) -> rotate -> translate)
        result = compute_procrustes_normalised_disparity(synth_coords, gt_coords, do_scaling=allow_scaling)

        X_trans = result['X_transformed']  # transformed synth points in GT coords (N,2)
        disp = result['normalized_disparity']
        scale = result['scale']
        rot_deg = result['rotation_deg']

        procrustes_df.loc[len(procrustes_df)] = {"step": step, "normalised_disparity": disp, "scale": scale,
                                                 "rotation": f"{rot_deg:.2f}°"}

        # hull polygons for plotting
        gt_hx, gt_hy = hull_polygon(gt_coords)
        tr_hx, tr_hy = hull_polygon(X_trans)

        # prepare traces that will go into this frame
        traces = [
            go.Scatter(x=gt_coords[:, 0], y=gt_coords[:, 1], mode="markers+text", marker=dict(size=7, color="#D55E00"),
                       text=corner_vowels, textposition="top center", textfont=dict(color="#D55E00"),
                       hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>", name="Vowels",
                       showlegend=False),
            go.Scatter(x=gt_hx, y=gt_hy, mode='lines', line=dict(width=2, color="#D55E00", dash="dot"),
                       hoverinfo="skip", name="Ground Truth Vowel Space Shape", showlegend=True),
            go.Scatter(x=synth_coords[:, 0], y=synth_coords[:, 1], mode="markers+text",
                       marker=dict(size=7, color="#0072B2"),
                       text=corner_vowels, textposition="top center", textfont=dict(color="#0072B2"),
                       hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>", name="Vowels",
                       showlegend=False, visible=True),
            go.Scatter(x=np.append(synth_coords[:, 0], synth_coords[:, 0][0]),
                       y=np.append(synth_coords[:, 1], synth_coords[:, 1][0]),
                       mode="lines", line=dict(width=1, color="#0072B2", dash="dot"),
                       name="Synthesised Vowel Space Shape", showlegend=True, visible=True),
            # Transformed synth hull
            go.Scatter(x=tr_hx, y=tr_hy, mode="lines", line=dict(width=2, color="#CC79A7"), fill="toself",
                       fillcolor="rgba(204, 121, 167, 0.15)", hoverinfo="skip",
                       name="Aligned Synthesised Vowel Space Shape", visible=True)]

        frames.append(go.Frame(data=traces, name=f"{step}"))

    fig.frames = frames

    if accent == "NZE":
        fig.update_layout(
            height=NZE_FIGURE_HEIGHT, width=NZE_FIGURE_WIDTH, template=PLOTLY_TEMPLATE,
            xaxis=NZE_X_AXIS, yaxis=NZE_Y_AXIS, legend=NZE_LEGEND, font=FONT,
            sliders=[dict(steps=[
                dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f.name,
                     method="animate") for f in frames], currentvalue=dict(prefix="Step: "), pad=dict(t=25))])
    elif accent == "GIE":
        fig.update_layout(
            height=GIE_FIGURE_HEIGHT, width=GIE_FIGURE_WIDTH, template=PLOTLY_TEMPLATE,
            xaxis=GIE_X_AXIS, yaxis=GIE_Y_AXIS, legend=GIE_LEGEND, font=FONT,
            sliders=[dict(steps=[
                dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f.name,
                     method="animate") for f in frames], currentvalue=dict(prefix="Step: "), pad=dict(t=25))])

    # save and return
    fig.write_html(output_html)


if __name__ == "__main__":
    csv_path = "CSVs/NZE-formants.csv"
    df = pd.read_csv(csv_path)
    corner_vowels = ["FLEECE", "THOUGHT", "START"]
    steps_ = [0, 1000, 3000, 7000, 10000, 16000, 20000, 28000]
    accent_ = "NZE"
    figure_path_ = "plots/NZE_procrustes_disparity.html"
    create_procrustes_animation(df, corner_vowels, step_prefixes=steps_, accent=accent_, allow_scaling=True,
                                output_html=figure_path_)

    csv_path = "CSVs/GIE-formants.csv"
    df = pd.read_csv(csv_path)
    corner_vowels = ["FLEECE", "GOOSE", "PALM"]
    steps_ = [0, 1000, 3000, 7000, 10000, 16000, 24000, 28000]
    accent_ = "GIE"
    figure_path_ = "plots/GIE_procrustes_disparity.html"
    create_procrustes_animation(df, corner_vowels, step_prefixes=steps_, accent=accent_, allow_scaling=True,
                                output_html=figure_path_)
