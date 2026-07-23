#!/usr/bin/env python3
"""Convert the concise bilingual Markdown speaking notes into an iPad-friendly PDF."""

import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msc-matplotlib")

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "docs/reporting/supervisor_progress_report_speaking_notes_concise_bilingual.md"
)
OUTPUT = (
    ROOT
    / "docs/reporting/supervisor_progress_report_speaking_notes_concise_bilingual.pdf"
)

PAGE_SIZE = (8.27, 11.69)  # A4 portrait
NAVY = "#17324D"
BLUE = "#2364AA"
TEAL = "#18A999"
ORANGE = "#F28E2B"
TEXT = "#25313C"
GREY = "#6B7785"
LIGHT_BLUE = "#EDF5FA"
LIGHT_ORANGE = "#FFF5E8"
LINE = "#D8DEE5"
WHITE = "#FFFFFF"

FONT_REGULAR = FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
FONT_BOLD = FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
)
rcParams["pdf.fonttype"] = 42


def clean_markdown(value: str) -> str:
    value = value.replace("`", "")
    value = value.replace("**", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_markdown(path: Path) -> list[tuple[str, str]]:
    """Parse the small Markdown subset used by the speaking notes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    elements: list[tuple[str, str]] = []
    index = 0

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            elements.append(("space", ""))
            index += 1
            continue

        if stripped == "---":
            elements.append(("pagebreak", ""))
            index += 1
            continue

        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                quote_line = lines[index].lstrip()[1:].strip()
                quote_lines.append(clean_markdown(quote_line))
                index += 1
            elements.append(("quote", "\n".join(quote_lines)))
            continue

        if stripped.startswith("### "):
            elements.append(("h3", clean_markdown(stripped[4:])))
        elif stripped.startswith("## "):
            elements.append(("h2", clean_markdown(stripped[3:])))
        elif stripped.startswith("# "):
            elements.append(("h1", clean_markdown(stripped[2:])))
        elif stripped == "中文提示：":
            elements.append(("note_label", stripped))
        elif stripped.startswith("- "):
            elements.append(("bullet", clean_markdown(stripped[2:])))
        else:
            elements.append(("body", clean_markdown(stripped)))
        index += 1

    return elements


class NotesRenderer:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self.page_number = 0
        self.fig = None
        self.ax = None
        self.y = 0.0
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.finish_page()

        self.page_number += 1
        self.fig = plt.figure(figsize=PAGE_SIZE, facecolor=WHITE)
        self.ax = self.fig.add_axes((0, 0, 1, 1))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.ax.add_patch(Rectangle((0, 0.962), 1, 0.038, facecolor=NAVY, edgecolor="none"))
        self.ax.text(
            0.075,
            0.981,
            "Concise Bilingual Speaking Notes",
            fontsize=8.5,
            color=WHITE,
            fontproperties=FONT_BOLD,
            va="center",
        )
        self.y = 0.925

    def finish_page(self):
        self.ax.plot([0.075, 0.925], [0.045, 0.045], color=LINE, lw=0.7)
        self.ax.text(
            0.075,
            0.025,
            "Pang Zhenkun | Supervisor Progress Report",
            fontsize=7.5,
            color=GREY,
            fontproperties=FONT_REGULAR,
            va="center",
        )
        self.ax.text(
            0.925,
            0.025,
            str(self.page_number),
            fontsize=8,
            color=GREY,
            fontproperties=FONT_REGULAR,
            ha="right",
            va="center",
        )
        self.pdf.savefig(self.fig, facecolor=WHITE)
        plt.close(self.fig)

    def ensure_space(self, height: float):
        if self.y - height < 0.075:
            self.new_page()

    def draw_wrapped_text(
        self,
        value: str,
        *,
        x: float,
        size: float,
        color: str,
        bold: bool,
        width: int,
        line_height: float,
        prefix: str = "",
    ):
        wrapped = textwrap.wrap(
            value,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        height = len(wrapped) * line_height
        self.ensure_space(height + 0.006)

        for line_index, line in enumerate(wrapped):
            line_prefix = prefix if line_index == 0 else "   " if prefix else ""
            self.ax.text(
                x,
                self.y,
                f"{line_prefix}{line}",
                fontsize=size,
                color=color,
                fontproperties=FONT_BOLD if bold else FONT_REGULAR,
                va="top",
            )
            self.y -= line_height
        return height

    def render(self, kind: str, value: str):
        if kind == "pagebreak":
            if self.y < 0.88:
                self.new_page()
            return

        if kind == "space":
            self.y -= 0.010
            return

        if kind == "h1":
            self.ensure_space(0.070)
            self.ax.text(
                0.075,
                self.y,
                value,
                fontsize=21,
                color=NAVY,
                fontproperties=FONT_BOLD,
                va="top",
            )
            self.y -= 0.065
            return

        if kind == "h2":
            self.ensure_space(0.055)
            self.ax.text(
                0.075,
                self.y,
                value,
                fontsize=15.5,
                color=BLUE,
                fontproperties=FONT_BOLD,
                va="top",
            )
            self.y -= 0.047
            return

        if kind == "h3":
            self.ensure_space(0.045)
            self.ax.text(
                0.075,
                self.y,
                value,
                fontsize=12.5,
                color=TEAL,
                fontproperties=FONT_BOLD,
                va="top",
            )
            self.y -= 0.038
            return

        if kind == "quote":
            paragraphs = value.split("\n")
            wrapped_lines: list[str] = []
            for paragraph in paragraphs:
                if paragraph:
                    wrapped_lines.extend(
                        textwrap.wrap(
                            paragraph,
                            width=76,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )
                else:
                    wrapped_lines.append("")

            line_height = 0.026
            box_height = len(wrapped_lines) * line_height + 0.024
            self.ensure_space(box_height + 0.012)
            box_y = self.y - box_height + 0.004
            self.ax.add_patch(
                FancyBboxPatch(
                    (0.075, box_y),
                    0.85,
                    box_height,
                    boxstyle="round,pad=0.006,rounding_size=0.012",
                    facecolor=LIGHT_BLUE,
                    edgecolor="none",
                )
            )
            self.ax.add_patch(
                Rectangle(
                    (0.075, box_y + 0.006),
                    0.007,
                    box_height - 0.012,
                    facecolor=TEAL,
                    edgecolor="none",
                )
            )
            text_y = self.y - 0.009
            for line in wrapped_lines:
                self.ax.text(
                    0.097,
                    text_y,
                    line,
                    fontsize=12.2,
                    color=NAVY,
                    fontproperties=FONT_REGULAR,
                    va="top",
                )
                text_y -= line_height
            self.y -= box_height + 0.014
            return

        if kind == "note_label":
            self.ensure_space(0.038)
            self.ax.add_patch(
                FancyBboxPatch(
                    (0.075, self.y - 0.026),
                    0.15,
                    0.032,
                    boxstyle="round,pad=0.003,rounding_size=0.009",
                    facecolor=LIGHT_ORANGE,
                    edgecolor="none",
                )
            )
            self.ax.text(
                0.087,
                self.y - 0.010,
                "中文提示",
                fontsize=10.5,
                color=ORANGE,
                fontproperties=FONT_BOLD,
                va="center",
            )
            self.y -= 0.041
            return

        if kind == "bullet":
            self.draw_wrapped_text(
                value,
                x=0.095,
                size=10.3,
                color=GREY,
                bold=False,
                width=48,
                line_height=0.022,
                prefix="• ",
            )
            self.y -= 0.004
            return

        self.draw_wrapped_text(
            value,
            x=0.075,
            size=10.5,
            color=TEXT,
            bold=False,
            width=72,
            line_height=0.023,
        )
        self.y -= 0.005

    def close(self):
        self.finish_page()
        self.fig = None


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Markdown source not found: {SOURCE}")

    elements = parse_markdown(SOURCE)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        renderer = NotesRenderer(pdf)
        for kind, value in elements:
            renderer.render(kind, value)
        renderer.close()

    print(f"Generated: {OUTPUT}")


if __name__ == "__main__":
    main()
