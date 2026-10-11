"""Conservative line-grid detector for scanned PDF table images.

Requires optional OpenCV + Tesseract. Only explicit ruled grids are recognized;
complex borderless tables are left as OCR text. Never fabricates empty cells.
"""
from __future__ import annotations

import logging
from PIL import Image
from .table_intelligence import table_markdown

log = logging.getLogger("koib.scanned_tables")


def _cluster(values: list[int], tolerance: int) -> list[int]:
    groups: list[list[int]] = []
    for v in sorted(values):
        if groups and abs(v - groups[-1][-1]) <= tolerance:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [round(sum(g) / len(g)) for g in groups]


def read_ruled_table(image: Image.Image, *, lang: str = "rus+eng", max_cells: int = 240) -> dict | None:
    """Find and OCR a drawn grid; return None on ambiguity or missing engines."""
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        log.info("Scanned table OCR disabled: cv2 / numpy / pytesseract missing")
        return None
    pix = np.asarray(image.convert("RGB"))
    height, width = pix.shape[:2]
    if width < 180 or height < 80:
        return None
    gray = cv2.cvtColor(pix, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, width // 20), 1)))
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, height // 20))))
    crosses = cv2.bitwise_and(horiz, vert)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(crosses, 8)
    pts = [(int(centroids[i][0]), int(centroids[i][1])) for i in range(1, count)
           if 1 <= stats[i][cv2.CC_STAT_AREA] <= 500]
    if len(pts) < 6:
        return None
    xs = _cluster([x for x, _ in pts], max(3, width // 300))
    ys = _cluster([y for _, y in pts], max(3, height // 300))
    ncols, nrows = len(xs) - 1, len(ys) - 1
    if ncols < 2 or nrows < 1 or ncols * nrows > max_cells:
        return None
    # Require an intersection at every grid crossing, not merely aligned text.
    tolerance = max(6, min(width, height) // 200)
    if any(not any(abs(px-x) <= tolerance and abs(py-y) <= tolerance for px,py in pts)
           for x in xs for y in ys):
        return None
    if min(xs[i + 1] - xs[i] for i in range(ncols)) < 28:
        return None
    if min(ys[i + 1] - ys[i] for i in range(nrows)) < 18:
        return None
    rows = []
    non_empty = 0
    for j in range(nrows):
        row = []
        for i in range(ncols):
            x0, x1 = xs[i] + 4, xs[i + 1] - 4
            y0, y1 = ys[j] + 3, ys[j + 1] - 3
            if x1 <= x0 or y1 <= y0:
                return None
            crop = image.crop((x0, y0, x1, y1)).convert("RGB")
            try:
                word = pytesseract.image_to_string(crop, lang=lang, config="--psm 7").strip()
            except Exception as exc:
                log.debug("Cell OCR failed: %s", exc)
                return None
            word = " ".join(word.split())
            if word:
                non_empty += 1
            row.append(word)
        rows.append(row)
    if non_empty < max(3, ncols):
        return None
    return {"text": table_markdown(rows), "num_rows": nrows, "num_cols": ncols,
            "ocr": True, "ocr_method": "ruled_grid", "ocr_cell_count": ncols * nrows,
            "confidence": "heuristic", "bbox_pixels": [xs[0], ys[0], xs[-1], ys[-1]]}
