# %%
from typing import List, Optional

from numpy import nan
from pandas import DataFrame, read_csv
from plotly.graph_objects import Figure, Frame, Scatter

from src.constants import *


# %%
class Coordinate:
    def __init__(self, vowel: str, f1: float, f2: float) -> None:
        self.vowel: str = vowel
        self.x: float = f2
        self.y: float = f1

    def __repr__(self) -> str:
        return f"({self.vowel}, {self.x}, {self.y})"


# %%
def get_vowel_coordinates(formant_df: DataFrame, f1_col_name: str, f2_col_name: str, vowel_col_name="vowel") -> list[
    Coordinate]:
    coordinates: list[Coordinate] = []
    for _, row in formant_df.iterrows():
        coordinates.append(Coordinate(vowel=row.get(vowel_col_name), f1=row.get(f1_col_name), f2=row.get(f2_col_name)))
    return coordinates


# %%
def cross_product(o: Coordinate, a: Coordinate, b: Coordinate) -> float:
    """
    2D cross product (also called the determinant) of the vectors OA=(a.x−o.x,a.y−o.y) and OB=(b.x−o.x,b.y−o.y).
    The sign of this value tells you the orientation of the turn from OA to OB:
    Positive → counterclockwise (left turn)\n
    Negative → clockwise (right turn)\n
    Zero → collinear (all three points lie on a straight line).\n
    In convex hull algorithms like Andrew’s monotone chain, cross product is used to decide whether the last step added makes a right turn (which breaks convexity). If so, pop the last point until the turn becomes left (convex again).
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def compute_convex_hull(coordinates: list[Coordinate]) -> list[Coordinate]:
    """
    Computes convex hull of coordinates using Andrew's monotone chain.
    Returns list of hull vertices in counter-clockwise order (no repeated first/last).
    """
    # sort by x then y
    _coordinates: list[Coordinate] = sorted(coordinates, key=lambda p: (p.x, p.y))
    # unique by coordinates
    uniq = []
    seen = set()
    for c in _coordinates:
        key = (c.x, c.y)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    _coordinates = uniq

    if len(_coordinates) <= 1:
        return _coordinates[:]

    # lower hull
    lower = []
    for c in _coordinates:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], c) <= 0:
            lower.pop()
        lower.append(c)

    # upper hull
    upper = []
    for c in reversed(_coordinates):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], c) <= 0:
            upper.pop()
        upper.append(c)

    # concatenate lower and upper, omit last point of each (it's repeated)
    hull = lower[:-1] + upper[:-1]
    return hull


# %%
def compute_area(coordinates: list[Coordinate]) -> float:
    """
    Computes the area of a polygon (triangle, quadrilateral, etc.) using the shoelace formula.
    """
    n: int = len(coordinates)
    if n < 3:
        return 0.0

    area: float = 0.0
    for i in range(n):
        x0, y0 = coordinates[i].x, coordinates[i].y
        x1, y1 = coordinates[(i + 1) % n].x, coordinates[(i + 1) % n].y
        area += (x0 * y1) - (x1 * y0)

    return 0.5 * abs(area)


# %%
# -------------------- Polygon clipping (Sutherland–Hodgman) --------------------
def _is_inside(p: Coordinate, edge_start: Coordinate, edge_end: Coordinate) -> bool:
    """
    Return True if point p is on the left side (or on) of the directed edge (edge_start -> edge_end).
    We assume clip polygon is CCW; being left-of the edge means inside.
    """
    return cross_product(edge_start, edge_end, p) >= 0.0


# %%
def _line_intersection(a1: Coordinate, a2: Coordinate, b1: Coordinate, b2: Coordinate, eps=1e-12) -> Optional[
    Coordinate]:
    """
    Compute intersection point of lines (a1-a2) and (b1-b2).
    Returns Coordinate with empty label, or None if lines are (nearly) parallel.
    Uses the standard 2D line intersection formula.
    """
    x1, y1 = a1.x, a1.y
    x2, y2 = a2.x, a2.y
    x3, y3 = b1.x, b1.y
    x4, y4 = b2.x, b2.y

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < eps:
        return None  # parallel (or nearly so)

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return Coordinate(vowel="", f1=py, f2=px)  # remember constructor maps x=f2, y=f1
    # note: here we pass px->f2 and py->f1 so x=px, y=py in the result


# %%
def sutherland_hodgman_clip(subject_polygon: List[Coordinate], clip_polygon: List[Coordinate]) -> List[Coordinate]:
    """
    Clip subject_polygon by clip_polygon (both lists of Coordinate). Returns the intersection polygon as a list of Coordinates.
    Works for convex clip polygons (and in general too, but convex is typical here).
    If there's no overlap, returns an empty list.
    """
    if not subject_polygon or not clip_polygon:
        return []

    output_list = subject_polygon[:]
    clip_len = len(clip_polygon)

    for i in range(clip_len):
        cp1 = clip_polygon[i]
        cp2 = clip_polygon[(i + 1) % clip_len]
        input_list = output_list
        output_list = []

        if not input_list:
            break

        s = input_list[-1]
        for e in input_list:
            e_inside = _is_inside(e, cp1, cp2)
            s_inside = _is_inside(s, cp1, cp2)

            if e_inside:
                if not s_inside:
                    # entering — compute intersection and add it
                    inter = _line_intersection(s, e, cp1, cp2)
                    if inter is not None:
                        output_list.append(inter)
                output_list.append(e)  # add current endpoint
            elif s_inside:
                # leaving — compute intersection and add it
                inter = _line_intersection(s, e, cp1, cp2)
                if inter is not None:
                    output_list.append(inter)
            s = e

    # filter tiny/degenerate points (optional)
    # Note: output_list may contain Coordinates with empty vowel labels
    return output_list


# %%
def compute_intersection_area(poly_a: List[Coordinate], poly_b: List[Coordinate]) -> float:
    """
    Compute area of intersection polygon between poly_a and poly_b.
    Both inputs should be lists of Coordinates ordered (CCW or CW) representing polygons.
    This function clips poly_a by poly_b (works correctly for convex poly_b) and returns area of result.
    To be symmetric, you could clip the smaller by the larger — here we clip A by B, but we'll call it with
    (truth_hull, synth_hull) and that's fine. If you want guaranteed symmetry, clip truth by synth (the intersection is commutative).
    """
    # We'll clip poly_a by poly_b (Sutherland–Hodgman). For safety, ensure both are CCW;
    # monotone-chain produced hulls are CCW so usually no reorientation required.
    intersection_poly = sutherland_hodgman_clip(subject_polygon=poly_a, clip_polygon=poly_b)
    if not intersection_poly:
        return 0.0
    return compute_area(intersection_poly)


# %%
def compute_vowel_space_overlap(formant_df: DataFrame, steps: list[int], output_csv_path: str) -> DataFrame:
    metric_df: DataFrame = DataFrame(columns=["step", "area", "overlap_area", "overlap_pct"])

    coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=formant_df, f1_col_name="truth_f1",
                                                          f2_col_name=f"truth_f2")
    truth_convex_hull: list[Coordinate] = compute_convex_hull(coordinates=coordinates)
    truth_convex_hull_area: float = compute_area(coordinates=truth_convex_hull)
    metric_df.loc[len(metric_df)] = {"step": "truth", "area": truth_convex_hull_area, "overlap_area": nan,
                                     "overlap_pct": None}

    for step in steps:
        coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=formant_df, f1_col_name=f"{step}_f1",
                                                              f2_col_name=f"{step}_f2")
        synthesised_convex_hull: list[Coordinate] = compute_convex_hull(coordinates=coordinates)
        synthesised_convex_hull_area: float = compute_area(coordinates=synthesised_convex_hull)

        # overlap: intersection area between truth hull and synthesised hull
        overlap_area: float = compute_intersection_area(truth_convex_hull, synthesised_convex_hull)
        overlap_pct: Optional[
            str] = f"{(overlap_area / truth_convex_hull_area) * 100:.2f}%" if truth_convex_hull_area != 0.0 else None

        metric_df.loc[len(metric_df)] = {"step": step, "area": synthesised_convex_hull_area,
                                         "overlap_area": overlap_area, "overlap_pct": overlap_pct}

    metric_df.to_csv(path_or_buf=output_csv_path, index=False)
    return metric_df


# %%
def plot_vowel_space_overlap(formant_df: DataFrame, all_vowels_formant_df: DataFrame, steps: list[int], accent: str,
                             figure_path: str) -> None:
    figure: Figure = Figure()
    for _ in range(5):
        figure.add_trace(trace=Scatter())

    frames: list[Frame] = []

    truth_vowel_coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=all_vowels_formant_df,
                                                                      f1_col_name="truth_f1", f2_col_name=f"truth_f2")
    coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=formant_df, f1_col_name="truth_f1",
                                                          f2_col_name=f"truth_f2")
    truth_convex_hull: list[Coordinate] = compute_convex_hull(coordinates=coordinates)

    frame_traces: list[Scatter] = [
        Scatter(x=[c.x for c in truth_vowel_coordinates],
                y=[c.y for c in truth_vowel_coordinates],
                mode="markers+text",
                marker=dict(size=7, color="#D55E00"),
                text=[c.vowel for c in truth_vowel_coordinates],
                textposition="top center",
                textfont=dict(color="#D55E00"),
                hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>",
                name="Ground Truth Corner Vowels"),
        Scatter(x=[c.x for c in truth_convex_hull] + [truth_convex_hull[0].x],
                y=[c.y for c in truth_convex_hull] + [truth_convex_hull[0].y],
                mode="lines",
                line=dict(width=2, color="#D55E00"),
                # fill="toself",
                # fillcolor="rgba(213, 94, 0, 0.15)",
                hoverinfo="skip",
                name="Ground Truth Vowel Space Shape"),
        Scatter(visible=False),
        Scatter(visible=False),
        Scatter(visible=False)]

    # frames.append(Frame(data=frame_traces, name="Ground Truth"))

    for step in steps:
        all_vowel_coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=all_vowels_formant_df,
                                                                        f1_col_name=f"{step}_f1",
                                                                        f2_col_name=f"{step}_f2")
        coordinates: list[Coordinate] = get_vowel_coordinates(formant_df=formant_df, f1_col_name=f"{step}_f1",
                                                              f2_col_name=f"{step}_f2")
        convex_hull: list[Coordinate] = compute_convex_hull(coordinates=coordinates)
        overlap_convex_hull: List[Coordinate] = sutherland_hodgman_clip(subject_polygon=truth_convex_hull,
                                                                        clip_polygon=convex_hull)

        frame_traces: list[Scatter] = [
            Scatter(x=[c.x for c in truth_vowel_coordinates],
                    y=[c.y for c in truth_vowel_coordinates],
                    mode="markers+text",
                    marker=dict(size=7, color="#D55E00"),
                    text=[c.vowel for c in all_vowel_coordinates],
                    textposition="top center",
                    textfont=dict(color="#D55E00"),
                    hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>",
                    name="Ground Truth Corner Vowels",
                    showlegend=False),
            Scatter(x=[c.x for c in truth_convex_hull] + [truth_convex_hull[0].x],
                    y=[c.y for c in truth_convex_hull] + [truth_convex_hull[0].y],
                    mode="lines",
                    line=dict(width=2, color="#D55E00", dash="dot"),
                    # fill="toself",
                    # fillcolor="rgba(213, 94, 0, 0.15)",
                    hoverinfo="skip",
                    name="Ground Truth Vowel Space Shape"),
            Scatter(x=[c.x for c in all_vowel_coordinates],
                    y=[c.y for c in all_vowel_coordinates],
                    mode="markers+text",
                    marker=dict(size=7, color="#0072B2"),
                    text=[c.vowel for c in all_vowel_coordinates],
                    textposition="top center",
                    textfont=dict(color="#0072B2"),
                    hovertemplate="%{text}<br>F1 = %{y}<br>F2 = %{x}<extra></extra>",
                    name="Synthesised Corner Vowels",
                    showlegend=False,
                    visible=True),
            Scatter(x=[c.x for c in convex_hull] + [convex_hull[0].x],
                    y=[c.y for c in convex_hull] + [convex_hull[0].y],
                    mode="lines",
                    line=dict(width=2, color="#0072B2", dash="dot"),
                    # fill="toself",
                    # fillcolor="rgba(0,114,178,0.15)",
                    hoverinfo="skip",
                    name="Synthesised Vowel Space Shape",
                    visible=True),
            Scatter(x=[c.x for c in overlap_convex_hull] + [overlap_convex_hull[0].x],
                    y=[c.y for c in overlap_convex_hull] + [overlap_convex_hull[0].y],
                    mode="lines",
                    line=dict(width=2, color="#CC79A7"),
                    fill="toself",
                    fillcolor="rgba(204, 121, 167, 0.15)",
                    hoverinfo="skip",
                    name="Vowel Space Overlap",
                    visible=True)]

        frames.append(Frame(data=frame_traces, name=f"{step}"))

    figure.frames = frames

    if accent == "NZE":
        figure.update_layout(
            height=NZE_FIGURE_HEIGHT, width=NZE_FIGURE_WIDTH, template=PLOTLY_TEMPLATE,
            xaxis=NZE_X_AXIS, yaxis=NZE_Y_AXIS, legend=NZE_LEGEND, font=FONT,
            sliders=[dict(steps=[
                dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f.name,
                     method="animate") for f in frames], currentvalue=dict(prefix="Step: "), pad=dict(t=25))])
    elif accent == "GIE":
        figure.update_layout(
            height=GIE_FIGURE_HEIGHT, width=GIE_FIGURE_WIDTH, template=PLOTLY_TEMPLATE,
            xaxis=GIE_X_AXIS, yaxis=GIE_Y_AXIS, legend=GIE_LEGEND, font=FONT,
            sliders=[dict(steps=[
                dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True), mode="immediate")], label=f.name,
                     method="animate") for f in frames], currentvalue=dict(prefix="Step: "), pad=dict(t=25))])

    figure.write_html(figure_path)


# %%
formant_csv_path_ = "CSVs/NZE-formants.csv"
formant_df_: DataFrame = read_csv(filepath_or_buffer=formant_csv_path_)
corner_vowels_ = ["FLEECE", "THOUGHT", "START"]
corner_vowel_formant_df_ = formant_df_[formant_df_["vowel"].isin(values=corner_vowels_)]
steps_ = [0, 1000, 3000, 7000, 10000, 16000, 20000, 28000]

metric_csv_path_ = "CSVs/NZE_vowel_space_overlap.csv"
compute_vowel_space_overlap(formant_df=corner_vowel_formant_df_, steps=steps_, output_csv_path=metric_csv_path_)

accent_ = "NZE"
figure_path_ = "plots/NZE_vowel_space_overlap.html"
plot_vowel_space_overlap(formant_df=corner_vowel_formant_df_, all_vowels_formant_df=formant_df_, steps=steps_,
                         accent=accent_, figure_path=figure_path_)
# %%
formant_csv_path_ = "CSVs/GIE-formants.csv"
formant_df_: DataFrame = read_csv(filepath_or_buffer=formant_csv_path_)
corner_vowels_ = ["FLEECE", "GOOSE", "PALM"]
corner_vowel_formant_df_ = formant_df_[formant_df_["vowel"].isin(values=corner_vowels_)]
steps_ = [0, 1000, 3000, 7000, 10000, 16000, 24000, 28000]

metric_csv_path_ = "CSVs/GIE_vowel_space_overlap.csv"
compute_vowel_space_overlap(formant_df=corner_vowel_formant_df_, steps=steps_, output_csv_path=metric_csv_path_)

accent_ = "GIE"
figure_path_ = "plots/GIE_vowel_space_overlap.html"
plot_vowel_space_overlap(formant_df=corner_vowel_formant_df_, all_vowels_formant_df=formant_df_, steps=steps_,
                         accent=accent_, figure_path=figure_path_)
