#!/usr/bin/env python3
"""Generate the three-page English supervisor progress report."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msc-matplotlib")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from generate_supervisor_progress_report import (
    BLUE,
    DARK,
    LIGHT_BLUE,
    LIGHT_GREY,
    LIGHT_ORANGE,
    LIGHT_TEAL,
    MID_GREY,
    NAVY,
    ORANGE,
    PAGE_SIZE,
    SAMPLE_IMAGE,
    TEAL,
    WHITE,
    make_page,
    metric_chip,
    pipeline_card,
    rounded_box,
    table_cell,
    text,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/reporting/supervisor_progress_report_2026-07-23_en.pdf"

FULL_RESULTS = [
    ("Traditional CV", "56.95%", "0.3360", "29.62°"),
    ("VLM + geometry", "73.33%", "0.4182", "14.81°"),
    (
        "VLM + CNN (5 runs)",
        "74.51% ± 1.38%",
        "0.4510 ± 0.0081",
        "16.49° ± 0.72°",
    ),
]


def add_header(ax, title, subtitle=None):
    ax.add_patch(Rectangle((0, 0.89), 1, 0.11, facecolor=NAVY, edgecolor="none"))
    text(ax, 0.055, 0.952, title, size=23, color=WHITE, bold=True, va="center")
    if subtitle:
        text(
            ax,
            0.945,
            0.952,
            subtitle,
            size=10,
            color="#D7E5F2",
            ha="right",
            va="center",
        )


def add_footer(ax, page):
    ax.plot([0.055, 0.945], [0.045, 0.045], color="#D8DEE5", lw=0.8)
    text(
        ax,
        0.055,
        0.026,
        "VLM-guided 2D Robotic Grasp Detection | Pang Zhenkun | 2026-07-23",
        size=7.8,
        color=MID_GREY,
        va="center",
    )
    text(ax, 0.945, 0.026, f"{page} / 3", size=8, color=MID_GREY, ha="right", va="center")


def draw_page_1(pdf):
    fig, ax = make_page()
    add_header(ax, "Project Overview", "Whole-project progress")

    text(ax, 0.055, 0.845, "RESEARCH QUESTION", size=9.5, color=TEAL, bold=True)
    text(
        ax,
        0.055,
        0.807,
        "Can a pretrained open-vocabulary VLM improve\n2D grasp rectangle detection as a localisation front end?",
        size=16,
        color=NAVY,
        bold=True,
        linespacing=1.18,
    )
    text(
        ax,
        0.055,
        0.716,
        "Core idea: keep the localisation front end fixed and compare geometric and learned grasp back ends.",
        size=9.7,
        color=MID_GREY,
    )

    metric_chip(ax, 0.055, 0.585, 0.145, "DATASET", "885 samples", BLUE, LIGHT_BLUE)
    metric_chip(ax, 0.215, 0.585, 0.145, "BOX COVERAGE", "885 / 885", TEAL, LIGHT_TEAL)
    metric_chip(ax, 0.375, 0.585, 0.185, "OVERLAP", "IoU ≥ 0.25", ORANGE, LIGHT_ORANGE)
    metric_chip(ax, 0.575, 0.585, 0.185, "ORIENTATION", "Error ≤ 30°", ORANGE, LIGHT_ORANGE)

    image_ax = fig.add_axes((0.79, 0.565, 0.155, 0.205))
    image_ax.imshow(mpimg.imread(SAMPLE_IMAGE))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_color("#C8D2DC")
        spine.set_linewidth(1.0)
    text(ax, 0.8675, 0.545, "Example localisation and grasp", size=7.4, color=MID_GREY, ha="center")

    text(
        ax,
        0.055,
        0.545,
        "Cornell rectangle metric: a prediction is correct if it matches any positive annotation on both criteria "
        "(Jiang et al., 2011; Lenz et al., 2015).",
        size=7.6,
        color=MID_GREY,
    )
    text(ax, 0.055, 0.495, "Three Experimental Pipelines", size=13, color=NAVY, bold=True)

    pipeline_card(
        ax,
        0.055,
        "1",
        "Traditional CV",
        "Full RGB image\n↓\nThresholding + contours\n↓\nMin-area rotated rectangle",
        BLUE,
        LIGHT_BLUE,
    )
    pipeline_card(
        ax,
        0.3725,
        "2",
        "VLM + geometry",
        "Grounding DINO box\n↓\nSegment within the box\n↓\nGeometric grasp rectangle",
        TEAL,
        LIGHT_TEAL,
    )
    pipeline_card(
        ax,
        0.69,
        "3",
        "VLM + CNN",
        "Grounding DINO box\n↓\n224 × 224 RGB crop\n↓\nRegress 6 grasp parameters",
        ORANGE,
        LIGHT_ORANGE,
    )
    add_footer(ax, 1)
    pdf.savefig(fig, facecolor=WHITE)
    plt.close(fig)


def draw_page_2(pdf):
    fig, ax = make_page()
    add_header(ax, "Experimental Results", "Results first")

    text(ax, 0.055, 0.845, "Full Cornell dataset (885 samples)", size=13, color=NAVY, bold=True)
    table_x = 0.055
    table_top = 0.802
    widths = [0.255, 0.205, 0.205, 0.235]
    headers = ["Method", "Success rate", "Mean best IoU", "Mean angle error"]
    x = table_x
    for width, header in zip(widths, headers):
        table_cell(ax, x, table_top - 0.066, width, 0.066, header, NAVY, WHITE, True, 10)
        x += width

    row_colors = [WHITE, "#F7FBFD", LIGHT_TEAL]
    for row_index, row in enumerate(FULL_RESULTS):
        y = table_top - 0.066 - (row_index + 1) * 0.078
        x = table_x
        for col_index, (width, value) in enumerate(zip(widths, row)):
            highlight = row_index == 2 and col_index in (1, 2)
            best_angle = row_index == 1 and col_index == 3
            table_cell(
                ax,
                x,
                y,
                width,
                0.078,
                value,
                row_colors[row_index],
                TEAL if highlight or best_angle else DARK,
                highlight or best_angle or col_index == 0,
                9.5,
            )
            x += width

    rounded_box(ax, 0.055, 0.336, 0.45, 0.17, LIGHT_BLUE)
    text(ax, 0.077, 0.476, "Unseen-object test set (85 samples)", size=11.5, color=NAVY, bold=True)
    bar_x = 0.205
    bar_w = 0.245
    text(ax, 0.077, 0.425, "VLM + geometry", size=8.8, color=DARK, va="center")
    ax.add_patch(Rectangle((bar_x, 0.411), bar_w * 0.753, 0.032, facecolor=BLUE, edgecolor="none"))
    text(ax, bar_x + bar_w * 0.753 + 0.012, 0.427, "75.3%", size=10, color=BLUE, bold=True, va="center")
    text(ax, 0.077, 0.371, "VLM + CNN", size=8.8, color=DARK, va="center")
    ax.add_patch(Rectangle((bar_x, 0.357), bar_w * 0.8235, 0.032, facecolor=TEAL, edgecolor="none"))
    text(ax, bar_x + bar_w * 0.8235 + 0.012, 0.373, "82.35% ± 4.53%", size=10, color=TEAL, bold=True, va="center")

    rounded_box(ax, 0.535, 0.336, 0.41, 0.17, LIGHT_ORANGE)
    text(ax, 0.557, 0.476, "Failure analysis", size=11.5, color=NAVY, bold=True)
    text(ax, 0.557, 0.424, "126 / 236", size=21, color=ORANGE, bold=True, va="center")
    text(
        ax,
        0.69,
        0.424,
        "failures have a correct angle;\nthe main error is position or size.",
        size=9.2,
        color=DARK,
        va="center",
    )
    text(ax, 0.557, 0.367, "→ This motivates a learned grasp back end.", size=8.8, color=MID_GREY, va="center")

    text(ax, 0.055, 0.288, "Key Findings", size=13, color=NAVY, bold=True)
    conclusions = [
        ("+16.38 percentage points", "Largest gain from VLM", "56.95% → 73.33%", BLUE, LIGHT_BLUE),
        ("Higher IoU & generalisation", "CNN improves geometry", "Test: 82.35% ± 4.53%", TEAL, LIGHT_TEAL),
        ("14.81°", "Geometry is best for angle", "Explicit priors remain useful", ORANGE, LIGHT_ORANGE),
    ]
    for index, (metric, title_value, note, color, background) in enumerate(conclusions):
        x = 0.055 + index * 0.305
        rounded_box(ax, x, 0.105, 0.28, 0.14, background)
        text(ax, x + 0.018, 0.213, metric, size=12, color=color, bold=True)
        text(ax, x + 0.018, 0.172, title_value, size=9.4, color=NAVY, bold=True)
        text(ax, x + 0.018, 0.135, note, size=8.2, color=MID_GREY)

    add_footer(ax, 2)
    pdf.savefig(fig, facecolor=WHITE)
    plt.close(fig)


def status_item(ax, x, y, title_value, detail):
    rounded_box(ax, x, y - 0.014, 0.028, 0.028, TEAL, radius=0.01)
    text(ax, x + 0.014, y, "✓", size=11, color=WHITE, bold=True, ha="center", va="center")
    text(ax, x + 0.042, y + 0.012, title_value, size=9.5, color=NAVY, bold=True, va="center")
    text(ax, x + 0.042, y - 0.018, detail, size=7.9, color=MID_GREY, va="center")


def numbered_item(ax, x, y, number, title_value, detail):
    rounded_box(ax, x, y - 0.017, 0.034, 0.034, NAVY, radius=0.012)
    text(ax, x + 0.017, y, str(number), size=10, color=WHITE, bold=True, ha="center", va="center")
    text(ax, x + 0.048, y + 0.012, title_value, size=9.3, color=NAVY, bold=True, va="center")
    text(ax, x + 0.048, y - 0.018, detail, size=7.7, color=MID_GREY, va="center")


def draw_page_3(pdf):
    fig, ax = make_page()
    add_header(ax, "Current Status and Next Steps", "Experiments complete; dissertation next")

    rounded_box(ax, 0.055, 0.46, 0.425, 0.375, LIGHT_TEAL)
    text(ax, 0.078, 0.798, "Completed", size=14, color=TEAL, bold=True)
    status_item(ax, 0.078, 0.738, "Three experimental pipelines", "Traditional CV, VLM + geometry, VLM + CNN")
    status_item(ax, 0.078, 0.665, "Full evaluation", "885 Cornell samples with one rectangle metric")
    status_item(ax, 0.078, 0.592, "Stability analysis", "Five CNN runs with different random seeds")
    status_item(ax, 0.078, 0.519, "Failure analysis", "236 failures categorised by IoU and angle")

    rounded_box(ax, 0.52, 0.46, 0.425, 0.375, LIGHT_BLUE)
    text(ax, 0.543, 0.798, "Next Steps", size=14, color=BLUE, bold=True)
    numbered_item(ax, 0.543, 0.738, 1, "Write the dissertation", "Introduction, Methodology, Results, Discussion")
    numbered_item(ax, 0.543, 0.665, 2, "Prepare result figures", "Method comparison, failure modes, examples")
    numbered_item(ax, 0.543, 0.592, 3, "Add per-sample analysis", "Focus on CNN position, size and angle errors")
    numbered_item(ax, 0.543, 0.519, 4, "Optional hybrid back end", "CNN for geometry; explicit prior for orientation")

    rounded_box(ax, 0.055, 0.238, 0.425, 0.17, LIGHT_ORANGE)
    text(ax, 0.078, 0.374, "Current Limitations", size=12, color=ORANGE, bold=True)
    text(
        ax,
        0.078,
        0.332,
        "• Offline 2D evaluation on Cornell only\n"
        "• No physical robot control or closed-loop grasping\n"
        "• 885/885 means a box was returned, not that every box was perfect",
        size=8.7,
        color=DARK,
        linespacing=1.5,
    )

    rounded_box(ax, 0.52, 0.238, 0.425, 0.17, LIGHT_GREY)
    text(ax, 0.543, 0.374, "Key Literature", size=12, color=NAVY, bold=True)
    text(
        ax,
        0.543,
        0.332,
        "1. Jiang et al. (2011): grasp rectangles and evaluation\n"
        "2. Lenz et al. (2015): 30° + 25% IoU success criterion\n"
        "3. Liu et al. (2023): Grounding DINO open-set detection\n"
        "4. Morrison et al. (2018): GG-CNN and double-angle encoding",
        size=7.6,
        color=DARK,
        linespacing=1.42,
    )

    rounded_box(ax, 0.055, 0.112, 0.89, 0.075, NAVY)
    text(
        ax,
        0.5,
        0.149,
        "Stage conclusion: VLM localisation is effective; the next focus is grasp back-end refinement and dissertation writing.",
        size=11.2,
        color=WHITE,
        bold=True,
        ha="center",
        va="center",
    )

    add_footer(ax, 3)
    pdf.savefig(fig, facecolor=WHITE)
    plt.close(fig)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_IMAGE.exists():
        raise FileNotFoundError(f"Report example image not found: {SAMPLE_IMAGE}")
    with PdfPages(OUTPUT) as pdf:
        draw_page_1(pdf)
        draw_page_2(pdf)
        draw_page_3(pdf)
    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
