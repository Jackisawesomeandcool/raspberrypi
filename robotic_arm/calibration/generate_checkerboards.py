#!/usr/bin/env python3
"""Generate the print-ready checkerboards used by this project."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw


DPI = 300
MM_PER_INCH = 25.4
SQUARE_MM = 25.0

PAGE_A4 = (297.0, 210.0)
PAGE_A3 = (420.0, 297.0)
PAGE_A2 = (594.0, 420.0)

SINGLE_BOARDS = (
    ("a4_8x6_25mm", PAGE_A4, 8, 6),
    ("a3_12x8_25mm", PAGE_A3, 12, 8),
    ("a2_16x12_25mm", PAGE_A2, 16, 12),
)

# The A2 seams pass through square centres, not checkerboard corners.
A2_TILES = (
    ("top_left", (0.0, 0.0, 187.5, 137.5)),
    ("top_right", (187.5, 0.0, 400.0, 137.5)),
    ("bottom_left", (0.0, 137.5, 187.5, 300.0)),
    ("bottom_right", (187.5, 137.5, 400.0, 300.0)),
)


def mm_to_px(value_mm: float) -> int:
    return round(value_mm / MM_PER_INCH * DPI)


def black_rectangles(
    cols: int,
    rows: int,
    crop: tuple[float, float, float, float],
) -> list[tuple[float, float, float, float]]:
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    rectangles = []
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 != 0:
                continue
            square_x0 = col * SQUARE_MM
            square_y0 = row * SQUARE_MM
            square_x1 = square_x0 + SQUARE_MM
            square_y1 = square_y0 + SQUARE_MM
            x0 = max(square_x0, crop_x0)
            y0 = max(square_y0, crop_y0)
            x1 = min(square_x1, crop_x1)
            y1 = min(square_y1, crop_y1)
            if x0 < x1 and y0 < y1:
                rectangles.append((x0, y0, x1, y1))
    return rectangles


def crop_mark_segments(
    origin_x: float,
    origin_y: float,
    width: float,
    height: float,
) -> tuple[tuple[float, float, float, float], ...]:
    mark = 5.0
    gap = 2.0
    x0, y0 = origin_x, origin_y
    x1, y1 = origin_x + width, origin_y + height
    return (
        (x0 - gap - mark, y0, x0 - gap, y0),
        (x0, y0 - gap - mark, x0, y0 - gap),
        (x1 + gap, y0, x1 + gap + mark, y0),
        (x1, y0 - gap - mark, x1, y0 - gap),
        (x0 - gap - mark, y1, x0 - gap, y1),
        (x0, y1 + gap, x0, y1 + gap + mark),
        (x1 + gap, y1, x1 + gap + mark, y1),
        (x1, y1 + gap, x1, y1 + gap + mark),
    )


def svg_text(
    page: tuple[float, float],
    cols: int,
    rows: int,
    crop: tuple[float, float, float, float],
    label: str | None,
) -> str:
    page_w, page_h = page
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    crop_w = crop_x1 - crop_x0
    crop_h = crop_y1 - crop_y0
    origin_x = (page_w - crop_w) / 2.0
    origin_y = (page_h - crop_h) / 2.0

    elements = [
        f'  <rect width="{page_w}" height="{page_h}" fill="#ffffff"/>',
        f'  <clipPath id="board"><rect x="{origin_x:.6f}" y="{origin_y:.6f}" '
        f'width="{crop_w:.6f}" height="{crop_h:.6f}"/></clipPath>',
        '  <g clip-path="url(#board)">',
    ]
    for x0, y0, x1, y1 in black_rectangles(cols, rows, crop):
        elements.append(
            f'    <rect x="{origin_x + x0 - crop_x0:.6f}" '
            f'y="{origin_y + y0 - crop_y0:.6f}" '
            f'width="{x1 - x0:.6f}" height="{y1 - y0:.6f}" fill="#000000"/>'
        )
    elements.append("  </g>")

    if label is not None:
        for x0, y0, x1, y1 in crop_mark_segments(
            origin_x, origin_y, crop_w, crop_h
        ):
            elements.append(
                f'  <line x1="{x0:.6f}" y1="{y0:.6f}" '
                f'x2="{x1:.6f}" y2="{y1:.6f}" '
                'stroke="#000000" stroke-width="0.25"/>'
            )
        elements.append(
            f'  <text x="{page_w / 2:.6f}" y="8" text-anchor="middle" '
            'font-family="sans-serif" font-size="4" fill="#000000">'
            f"{escape(label)}</text>"
        )

    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg"',
            f'     width="{page_w}mm" height="{page_h}mm"',
            f'     viewBox="0 0 {page_w} {page_h}">',
            *elements,
            "</svg>",
            "",
        ]
    )


def save_png(
    path: Path,
    page: tuple[float, float],
    cols: int,
    rows: int,
    crop: tuple[float, float, float, float],
    label: str | None,
) -> None:
    page_px = (mm_to_px(page[0]), mm_to_px(page[1]))
    crop_x0, crop_y0, crop_x1, crop_y1 = crop
    crop_w_px = mm_to_px(crop_x1 - crop_x0)
    crop_h_px = mm_to_px(crop_y1 - crop_y0)
    origin_x = (page_px[0] - crop_w_px) // 2
    origin_y = (page_px[1] - crop_h_px) // 2

    image = Image.new("L", page_px, 255)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1 in black_rectangles(cols, rows, crop):
        draw.rectangle(
            (
                origin_x + mm_to_px(x0 - crop_x0),
                origin_y + mm_to_px(y0 - crop_y0),
                origin_x + mm_to_px(x1 - crop_x0) - 1,
                origin_y + mm_to_px(y1 - crop_y0) - 1,
            ),
            fill=0,
        )

    if label is not None:
        for x0, y0, x1, y1 in crop_mark_segments(
            0.0,
            0.0,
            crop_x1 - crop_x0,
            crop_y1 - crop_y0,
        ):
            draw.line(
                (
                    origin_x + mm_to_px(x0),
                    origin_y + mm_to_px(y0),
                    origin_x + mm_to_px(x1),
                    origin_y + mm_to_px(y1),
                ),
                fill=0,
                width=max(1, mm_to_px(0.25)),
            )
        draw.text(
            (page_px[0] // 2, mm_to_px(5.0)),
            label,
            fill=0,
            anchor="ma",
        )

    image.save(path, dpi=(DPI, DPI), optimize=True)


def write_board(
    output_dir: Path,
    name: str,
    page: tuple[float, float],
    cols: int,
    rows: int,
    crop: tuple[float, float, float, float],
    label: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{name}.svg"
    png_path = output_dir / f"{name}.png"
    svg_path.write_text(
        svg_text(page, cols, rows, crop, label),
        encoding="utf-8",
    )
    save_png(png_path, page, cols, rows, crop, label)
    print(svg_path)
    print(png_path)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "checkerboards"
    for name, page, cols, rows in SINGLE_BOARDS:
        write_board(
            output_dir,
            name,
            page,
            cols,
            rows,
            (0.0, 0.0, cols * SQUARE_MM, rows * SQUARE_MM),
        )

    tile_dir = output_dir / "a2_tiles"
    for name, crop in A2_TILES:
        write_board(
            tile_dir,
            name,
            PAGE_A4,
            16,
            12,
            crop,
            label=f"A2 TILE {name.upper()} - PRINT 100%",
        )


if __name__ == "__main__":
    main()
