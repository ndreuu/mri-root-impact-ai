"""Source-frame DICOM LPS geometry; no registration or anatomical confirmation."""

import math


LEVELS = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")


def _vector(value, count):
    try:
        result = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Geometry requires finite numeric vectors") from None
    if len(result) != count or not all(math.isfinite(v) for v in result):
        raise ValueError("Geometry requires finite numeric vectors")
    return result


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _axes(frame):
    position = _vector(frame["position"], 3)
    orientation = _vector(frame["orientation"], 6)
    spacing = _vector(frame["spacing"], 2)
    c, r = orientation[:3], orientation[3:]
    if (min(spacing) <= 0 or abs(dot(c, c) - 1) > 1e-3
            or abs(dot(r, r) - 1) > 1e-3 or abs(dot(c, r)) > 1e-3):
        raise ValueError("DICOM directions must be orthonormal with positive spacing")
    n = [c[1] * r[2] - c[2] * r[1], c[2] * r[0] - c[0] * r[2], c[0] * r[1] - c[1] * r[0]]
    norm = math.sqrt(dot(n, n))
    return position, [v * spacing[1] for v in c], [v * spacing[0] for v in r], [v / norm for v in n]


def pixel_to_lps(frame, x, y):
    x, y = _vector([x, y], 2)
    p, c, r, _ = _axes(frame)
    return [p[i] + c[i] * x + r[i] * y for i in range(3)]


def lps_to_pixel(frame, point):
    point = _vector(point, 3)
    p, c, r, n = _axes(frame)
    delta = [v - o for v, o in zip(point, p)]
    cc, rr, cr = dot(c, c), dot(r, r), dot(c, r)
    denominator = cc * rr - cr * cr
    x = (dot(delta, c) * rr - dot(delta, r) * cr) / denominator
    y = (dot(delta, r) * cc - dot(delta, c) * cr) / denominator
    return x, y, dot(delta, n)


def segment_plane_intersection(frame, a, b, tolerance=1e-6):
    a, b = _vector(a, 3), _vector(b, 3)
    p, _, _, n = _axes(frame)
    da = dot([v - o for v, o in zip(a, p)], n)
    db = dot([v - o for v, o in zip(b, p)], n)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Plane tolerance must be finite nonnegative millimeters")
    return min(da, db) <= tolerance and max(da, db) >= -tolerance


def level_segments(points):
    if not isinstance(points, dict) or set(points) != set(LEVELS):
        raise ValueError("All five ordered native disc points are required")
    p = [_vector(points[level], 3) for level in LEVELS]
    if any(a[2] - b[2] < 5 for a, b in zip(p, p[1:])):
        raise ValueError("Native disc points must descend by at least 5 mm; numbering unresolved")
    mids = [[(a[i] + b[i]) / 2 for i in range(3)] for a, b in zip(p, p[1:])]
    boundaries = [[2 * p[0][i] - mids[0][i] for i in range(3)]] + mids
    boundaries.append([2 * p[-1][i] - mids[-1][i] for i in range(3)])
    return {level: [boundaries[i], boundaries[i + 1]] for i, level in enumerate(LEVELS)}


def crop_box(frame, norm_xy, size_mm, offset_mm):
    x, y = _vector(norm_xy, 2)
    h, w = _vector(size_mm, 2)
    dy, dx = _vector(offset_mm, 2)
    _axes(frame)
    if not (0 <= x < 1 and 0 <= y < 1) or min(h, w) <= 0:
        raise ValueError("Invalid normalized crop center or physical crop size")
    rows, columns = frame["rows"], frame["columns"]
    if type(rows) is not int or type(columns) is not int or min(rows, columns) < 1:
        raise ValueError("Invalid source matrix")
    row_spacing, col_spacing = frame["spacing"]
    cx, cy = x * columns, y * rows
    left, right = round(cx - w / col_spacing / 2 + dx / col_spacing), round(cx + w / col_spacing / 2 + dx / col_spacing)
    top, bottom = round(cy - h / row_spacing / 2 + dy / row_spacing), round(cy + h / row_spacing / 2 + dy / row_spacing)
    if left < 0 or top < 0 or right >= columns or bottom >= rows or right <= left or bottom <= top:
        raise ValueError("Physical ROI is outside the source image or empty")
    return [left, top, right, bottom]


def box_corners_lps(frame, box):
    left, top, right, bottom = box
    return [pixel_to_lps(frame, x - .5, y - .5)
            for x, y in [(left, top), (right, top), (right, bottom), (left, bottom)]]
