#!/usr/bin/env python3
"""生成三页中文导师项目进展汇报 PDF。"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msc-matplotlib")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/reporting/supervisor_progress_report_2026-07-23.pdf"
SAMPLE_IMAGE = (
    ROOT
    / "data/processed/vlm/grasp/visualizations/success/01_pcd0100.png"
)

PAGE_SIZE = (11.69, 8.27)  # A4 landscape
NAVY = "#17324D"
BLUE = "#2364AA"
TEAL = "#18A999"
ORANGE = "#F28E2B"
LIGHT_BLUE = "#EAF3FA"
LIGHT_TEAL = "#E8F7F4"
LIGHT_ORANGE = "#FFF2E3"
LIGHT_GREY = "#F4F6F8"
MID_GREY = "#6B7785"
DARK = "#1F2933"
WHITE = "#FFFFFF"

FONT_REGULAR = FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
)
FONT_BOLD = FontProperties(
    fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
)
rcParams["pdf.fonttype"] = 42

FULL_RESULTS = [
    ("传统 CV", "56.95%", "0.3360", "29.62°"),
    ("VLM + 几何", "73.33%", "0.4182", "14.81°"),
    (
        "VLM + CNN（5 次）",
        "74.51% ± 1.38%",
        "0.4510 ± 0.0081",
        "16.49° ± 0.72°",
    ),
]


def make_page():
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=WHITE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def rounded_box(ax, x, y, width, height, color=WHITE, edge=None, radius=0.018):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=color,
        edgecolor=edge or color,
        linewidth=1.0,
    )
    ax.add_patch(patch)
    return patch


def text(
    ax,
    x,
    y,
    value,
    size=12,
    color=DARK,
    bold=False,
    ha="left",
    va="top",
    linespacing=1.35,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontproperties=FONT_BOLD if bold else FONT_REGULAR,
        ha=ha,
        va=va,
        linespacing=linespacing,
    )


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
        "VLM 引导的二维机器人抓取检测｜Pang Zhenkun｜2026-07-23",
        size=7.8,
        color=MID_GREY,
        va="center",
    )
    text(ax, 0.945, 0.026, f"{page} / 3", size=8, color=MID_GREY, ha="right", va="center")


def metric_chip(ax, x, y, width, title, value, color, background):
    rounded_box(ax, x, y, width, 0.098, background)
    text(ax, x + 0.018, y + 0.071, title, size=8.5, color=MID_GREY)
    text(ax, x + 0.018, y + 0.038, value, size=16, color=color, bold=True, va="center")


def pipeline_card(ax, x, number, title, steps, color, background):
    y = 0.13
    width = 0.255
    height = 0.24
    rounded_box(ax, x, y, width, height, background)
    rounded_box(ax, x + 0.018, y + 0.174, 0.045, 0.045, color, radius=0.014)
    text(
        ax,
        x + 0.0405,
        y + 0.196,
        number,
        size=11,
        color=WHITE,
        bold=True,
        ha="center",
        va="center",
    )
    text(ax, x + 0.076, y + 0.207, title, size=12.5, color=NAVY, bold=True)
    text(ax, x + 0.025, y + 0.145, steps, size=9.5, color=DARK, linespacing=1.55)


def draw_page_1(pdf):
    fig, ax = make_page()
    add_header(ax, "项目概览", "整个项目进展")

    text(
        ax,
        0.055,
        0.845,
        "研究问题",
        size=10,
        color=TEAL,
        bold=True,
    )
    text(
        ax,
        0.055,
        0.806,
        "预训练开放词汇 VLM 能否作为目标定位前端，提升二维抓取矩形检测？",
        size=18,
        color=NAVY,
        bold=True,
    )
    text(
        ax,
        0.055,
        0.755,
        "核心思路：保持定位前端一致，对比传统几何后端与学习式 CNN 后端。",
        size=10.5,
        color=MID_GREY,
    )

    metric_chip(ax, 0.055, 0.605, 0.145, "数据集", "885 个样本", BLUE, LIGHT_BLUE)
    metric_chip(ax, 0.215, 0.605, 0.145, "定位覆盖", "885 / 885", TEAL, LIGHT_TEAL)
    metric_chip(ax, 0.375, 0.605, 0.185, "成功判定", "IoU ≥ 0.25", ORANGE, LIGHT_ORANGE)
    metric_chip(ax, 0.575, 0.605, 0.185, "角度条件", "误差 ≤ 30°", ORANGE, LIGHT_ORANGE)

    image_ax = fig.add_axes((0.79, 0.585, 0.155, 0.205))
    image_ax.imshow(mpimg.imread(SAMPLE_IMAGE))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    for spine in image_ax.spines.values():
        spine.set_color("#C8D2DC")
        spine.set_linewidth(1.0)
    text(ax, 0.8675, 0.565, "VLM 定位与抓取预测示例", size=7.5, color=MID_GREY, ha="center")

    text(ax, 0.055, 0.525, "三条实验流程", size=13, color=NAVY, bold=True)
    pipeline_card(
        ax,
        0.055,
        "1",
        "传统 CV 基线",
        "完整 RGB 图像\n↓\n阈值分割与轮廓\n↓\n最小面积旋转矩形",
        BLUE,
        LIGHT_BLUE,
    )
    pipeline_card(
        ax,
        0.3725,
        "2",
        "VLM + 几何",
        "Grounding DINO 定位\n↓\n定位区域内分割\n↓\n几何抓取矩形",
        TEAL,
        LIGHT_TEAL,
    )
    pipeline_card(
        ax,
        0.69,
        "3",
        "VLM + CNN",
        "Grounding DINO 定位\n↓\n224 × 224 RGB 裁剪\n↓\n回归 6 个抓取参数",
        ORANGE,
        LIGHT_ORANGE,
    )
    add_footer(ax, 1)
    pdf.savefig(fig, facecolor=WHITE)
    plt.close(fig)


def table_cell(ax, x, y, width, height, value, color=WHITE, text_color=DARK, bold=False, size=10):
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            facecolor=color,
            edgecolor="#D8DEE5",
            linewidth=0.7,
        )
    )
    text(
        ax,
        x + width / 2,
        y + height / 2,
        value,
        size=size,
        color=text_color,
        bold=bold,
        ha="center",
        va="center",
    )


def draw_page_2(pdf):
    fig, ax = make_page()
    add_header(ax, "实验结果", "结果优先")

    text(ax, 0.055, 0.845, "完整 Cornell 数据集（885 个样本）", size=13, color=NAVY, bold=True)
    table_x = 0.055
    table_top = 0.802
    widths = [0.255, 0.205, 0.205, 0.235]
    headers = ["方法", "成功率", "平均最佳 IoU", "平均角度误差"]
    x = table_x
    for width, header in zip(widths, headers):
        table_cell(ax, x, table_top - 0.066, width, 0.066, header, NAVY, WHITE, True, 10)
        x += width

    row_colors = [WHITE, "#F7FBFD", LIGHT_TEAL]
    for row_index, row in enumerate(FULL_RESULTS):
        y = table_top - 0.066 - (row_index + 1) * 0.078
        x = table_x
        for col_index, (width, value) in enumerate(zip(widths, row)):
            is_highlight = row_index == 2 and col_index in (1, 2)
            is_best_angle = row_index == 1 and col_index == 3
            table_cell(
                ax,
                x,
                y,
                width,
                0.078,
                value,
                row_colors[row_index],
                TEAL if is_highlight or is_best_angle else DARK,
                is_highlight or is_best_angle or col_index == 0,
                10 if row_index < 2 else 9.2,
            )
            x += width

    rounded_box(ax, 0.055, 0.336, 0.45, 0.17, LIGHT_BLUE)
    text(ax, 0.077, 0.476, "未见物体测试集（85 个样本）", size=11.5, color=NAVY, bold=True)
    bar_x = 0.205
    bar_w = 0.245
    text(ax, 0.077, 0.425, "VLM + 几何", size=9, color=DARK, va="center")
    ax.add_patch(Rectangle((bar_x, 0.411), bar_w * 0.753, 0.032, facecolor=BLUE, edgecolor="none"))
    text(ax, bar_x + bar_w * 0.753 + 0.012, 0.427, "75.3%", size=10, color=BLUE, bold=True, va="center")
    text(ax, 0.077, 0.371, "VLM + CNN", size=9, color=DARK, va="center")
    ax.add_patch(Rectangle((bar_x, 0.357), bar_w * 0.8235, 0.032, facecolor=TEAL, edgecolor="none"))
    text(ax, bar_x + bar_w * 0.8235 + 0.012, 0.373, "82.35% ± 4.53%", size=10, color=TEAL, bold=True, va="center")

    rounded_box(ax, 0.535, 0.336, 0.41, 0.17, LIGHT_ORANGE)
    text(ax, 0.557, 0.476, "失败案例分析", size=11.5, color=NAVY, bold=True)
    text(ax, 0.557, 0.424, "126 / 236", size=21, color=ORANGE, bold=True, va="center")
    text(ax, 0.69, 0.424, "失败样本角度正确，\n主要问题是位置或尺寸不准。", size=10, color=DARK, va="center")
    text(ax, 0.557, 0.367, "→ 这直接支持引入学习式抓取后端。", size=9.5, color=MID_GREY, va="center")

    text(ax, 0.055, 0.288, "关键结论", size=13, color=NAVY, bold=True)
    conclusions = [
        ("+16.38 个百分点", "VLM 定位带来最大提升", "56.95% → 73.33%", BLUE, LIGHT_BLUE),
        ("更高 IoU 与泛化", "CNN 改善位置和尺寸", "测试集 82.35% ± 4.53%", TEAL, LIGHT_TEAL),
        ("14.81°", "几何后端角度更准", "显式方向先验仍然有效", ORANGE, LIGHT_ORANGE),
    ]
    for index, (metric, title_value, note, color, background) in enumerate(conclusions):
        x = 0.055 + index * 0.305
        rounded_box(ax, x, 0.105, 0.28, 0.14, background)
        text(ax, x + 0.018, 0.213, metric, size=14, color=color, bold=True)
        text(ax, x + 0.018, 0.172, title_value, size=10, color=NAVY, bold=True)
        text(ax, x + 0.018, 0.135, note, size=8.5, color=MID_GREY)

    add_footer(ax, 2)
    pdf.savefig(fig, facecolor=WHITE)
    plt.close(fig)


def status_item(ax, x, y, title_value, detail):
    rounded_box(ax, x, y - 0.014, 0.028, 0.028, TEAL, radius=0.01)
    text(ax, x + 0.014, y, "✓", size=11, color=WHITE, bold=True, ha="center", va="center")
    text(ax, x + 0.042, y + 0.012, title_value, size=10, color=NAVY, bold=True, va="center")
    text(ax, x + 0.042, y - 0.018, detail, size=8.5, color=MID_GREY, va="center")


def numbered_item(ax, x, y, number, title_value, detail):
    rounded_box(ax, x, y - 0.017, 0.034, 0.034, NAVY, radius=0.012)
    text(ax, x + 0.017, y, str(number), size=10, color=WHITE, bold=True, ha="center", va="center")
    text(ax, x + 0.048, y + 0.012, title_value, size=10, color=NAVY, bold=True, va="center")
    text(ax, x + 0.048, y - 0.018, detail, size=8.5, color=MID_GREY, va="center")


def draw_page_3(pdf):
    fig, ax = make_page()
    add_header(ax, "当前状态与下一步", "实验完成，转向论文写作")

    rounded_box(ax, 0.055, 0.46, 0.425, 0.375, LIGHT_TEAL)
    text(ax, 0.078, 0.798, "已完成", size=14, color=TEAL, bold=True)
    status_item(ax, 0.078, 0.738, "三条实验流程", "传统 CV、VLM + 几何、VLM + CNN")
    status_item(ax, 0.078, 0.665, "全量评估", "Cornell 885 个样本，统一矩形指标")
    status_item(ax, 0.078, 0.592, "稳定性验证", "CNN 使用 5 个随机种子重复实验")
    status_item(ax, 0.078, 0.519, "失败案例分析", "236 个失败样本按 IoU 与角度分类")

    rounded_box(ax, 0.52, 0.46, 0.425, 0.375, LIGHT_BLUE)
    text(ax, 0.543, 0.798, "下一步", size=14, color=BLUE, bold=True)
    numbered_item(ax, 0.543, 0.738, 1, "完成论文主体", "Introduction、Methodology、Results、Discussion")
    numbered_item(ax, 0.543, 0.665, 2, "整理实验图表", "方法对比、失败分布、成功与失败示例")
    numbered_item(ax, 0.543, 0.592, 3, "补充逐样本分析", "重点分析 CNN 的位置、尺寸与角度误差")
    numbered_item(ax, 0.543, 0.519, 4, "可选混合后端", "CNN 回归位置与尺寸，几何方法提供角度先验")

    rounded_box(ax, 0.055, 0.238, 0.425, 0.17, LIGHT_ORANGE)
    text(ax, 0.078, 0.374, "当前局限", size=12, color=ORANGE, bold=True)
    text(
        ax,
        0.078,
        0.332,
        "• 目前只在 Cornell 数据集上进行离线二维评估\n"
        "• 未执行真实机械臂控制或闭环抓取\n"
        "• 885 / 885 表示当前设置下有检测输出，不等于定位框完全准确",
        size=9.3,
        color=DARK,
        linespacing=1.5,
    )

    rounded_box(ax, 0.52, 0.238, 0.425, 0.17, LIGHT_GREY)
    text(ax, 0.543, 0.374, "关键论文依据", size=12, color=NAVY, bold=True)
    text(
        ax,
        0.543,
        0.332,
        "1. Liu et al. (2023), Grounding DINO：开放集目标检测\n"
        "2. Morrison et al. (2018), GG-CNN：实时生成式抓取预测，\n"
        "   并使用 sin(2θ)、cos(2θ) 表示平行夹爪角度",
        size=8.8,
        color=DARK,
        linespacing=1.5,
    )

    rounded_box(ax, 0.055, 0.112, 0.89, 0.075, NAVY)
    text(
        ax,
        0.5,
        0.149,
        "阶段结论：VLM 已有效解决目标区域定位；下一阶段重点是抓取后端优化与论文整理。",
        size=13,
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
        raise FileNotFoundError(f"找不到汇报示例图：{SAMPLE_IMAGE}")
    with PdfPages(OUTPUT) as pdf:
        draw_page_1(pdf)
        draw_page_2(pdf)
        draw_page_3(pdf)
    print(f"已生成：{OUTPUT}")


if __name__ == "__main__":
    main()
