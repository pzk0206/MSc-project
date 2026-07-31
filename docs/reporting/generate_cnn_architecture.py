"""Generate the dissertation's lightweight CNN architecture figure as vector PDF."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "uog_dissertation_outline"
    / "images"
    / "cnn_architecture.pdf"
)


def main() -> None:
    stages = [
        ("RGB crop", "3×224×224", "#D9EAF7", 1.00),
        ("Block 1", "Conv 5×5, s2\n32 + BN + ReLU\nMaxPool\n32×56×56", "#B9D8EE", 1.20),
        ("Block 2", "Conv 3×3\n64 + BN + ReLU\nMaxPool\n64×28×28", "#B9D8EE", 1.20),
        ("Block 3", "Conv 3×3\n128 + BN + ReLU\nMaxPool\n128×14×14", "#B9D8EE", 1.20),
        ("Block 4", "Conv 3×3\n256 + BN + ReLU\nMaxPool\n256×7×7", "#B9D8EE", 1.20),
        ("Adaptive GAP", "256×7×7 → 256", "#CEE8D1", 1.20),
        ("FC 128", "ReLU\nDropout 0.3", "#F7D6B5", 1.00),
        ("FC 64", "ReLU\nDropout 0.2", "#F7D6B5", 1.00),
        (
            "Linear 6",
            "cx, cy, w, h\nsin(2θ), cos(2θ)",
            "#F4C7C3",
            1.45,
        ),
    ]

    gap = 0.30
    height = 2.20
    x_positions = []
    x = 0.25
    for _, _, _, width in stages:
        x_positions.append(x)
        x += width + gap

    fig, ax = plt.subplots(figsize=(17.2, 4.1))
    ax.set_xlim(0, x + 0.05)
    ax.set_ylim(0, 3.75)
    ax.axis("off")

    for index, ((title, detail, colour, width), stage_x) in enumerate(
        zip(stages, x_positions, strict=True)
    ):
        box = FancyBboxPatch(
            (stage_x, 1.05),
            width,
            height,
            boxstyle="round,pad=0.035,rounding_size=0.07",
            linewidth=1.15,
            edgecolor="#34495E",
            facecolor=colour,
        )
        ax.add_patch(box)
        ax.text(
            stage_x + width / 2,
            2.87,
            title,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
        )
        ax.text(
            stage_x + width / 2,
            2.05,
            detail,
            ha="center",
            va="center",
            fontsize=8.8,
            linespacing=1.35,
        )

        if index < len(stages) - 1:
            next_x = x_positions[index + 1]
            arrow = FancyArrowPatch(
                (stage_x + width + 0.025, 2.15),
                (next_x - 0.025, 2.15),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color="#34495E",
            )
            ax.add_patch(arrow)

    ax.text(
        (x + 0.25) / 2,
        0.55,
        "432,454 trainable parameters. Lightweight regression baseline designed "
        "for this study; not proposed as a novel architecture.",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="semibold",
        color="#263238",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
