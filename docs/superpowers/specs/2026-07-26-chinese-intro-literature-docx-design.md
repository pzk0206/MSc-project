# 精简中文 Introduction 与 Literature Review DOCX 设计

日期：2026-07-26

## 目标

生成一份可直接用 Word 阅读和批注的精简中文初稿，覆盖毕业论文的
Introduction 与 Literature Review。中文稿用于帮助作者理解、修改和确认论证，
后续再翻译为英文 LaTeX 正文；本次不直接替换现有英文论文。

正式输出：

```text
docs/drafts/introduction_literature_review_zh.docx
```

## 篇幅与风格

- 总篇幅约 3000–4000 中文字。
- 采用正式但清晰的学术中文，避免堆砌术语和过长背景。
- 每一节围绕一个明确问题展开，使用“论点—证据—与本项目关系”的结构。
- 使用 Word 一级、二级标题和普通正文样式，便于逐段批注和未来翻译。
- 不保留 `TODO`、占位符或未经解释的英文缩写。

## Introduction 内容

### 背景与动机

说明机器人抓取检测、二维抓取矩形、视觉定位与开放词汇感知的关系。指出语义
定位和抓取几何是不同问题，并说明把 VLM 定位作为可替换前端的研究价值。

### 研究问题

使用三个问题组织后续章节：

1. Grounding DINO 开放词汇定位是否能改善传统整图几何抓取流程？
2. 在共同 VLM crop 下，学习式 CNN 后端与几何后端分别表现出什么优势？
3. 在相同 image-wise manifest 下，多头回归是否优于单头回归？

### 研究目标与任务

概括数据解析、传统 CV 基线、VLM 定位、几何后端、单头/多头 CNN、确定性重复
实验、image-wise 五折、失败分析和结果审计。

### 项目范围

明确项目是 Cornell 上的离线 RGB 二维感知研究，不包括深度融合、机器人控制、
碰撞检测或物理抓取成功率。Object-wise 因缺少权威实例映射而不作结果声明。

### 论文结构

简要说明 Literature Review、Methodology、Results、Discussion 和 Conclusion
的职责。

## Literature Review 内容

### 机器人抓取检测

说明抓取检测从物体定位到可执行抓取表示的任务结构，以及视觉感知的重要性。

### 二维抓取矩形与 Cornell

回顾 Jiang、Lenz、Redmon and Angelova 对矩形表示、候选评分和直接回归的
发展；解释 Cornell rectangle metric 和 image-wise/object-wise 的差别。

### 传统 CV 与深度学习

比较阈值、轮廓和几何规则的可解释性与脆弱性；比较全局 CNN、ResNet 和密集
生成式网络的表达能力、数据依赖与输入模态差异。涵盖 GG-CNN、GR-ConvNet
和 Gaussian-guided 方法，但不构造不公平排行榜。

### VLM 与开放词汇定位

从 CLIP 的视觉语言对齐过渡到 Grounding DINO 的文本条件检测，并联系语言驱动
抓取研究。明确 Grounding DINO 提供的是定位框，不直接解决抓取几何。

### 轻量适配

简要介绍 LoRA/QLoRA 的参数高效适配思想，并说明本项目没有把微调作为核心
实验，以避免在一个月范围内同时扩大模型、输入和评价协议。

### 研究缺口

将缺口限定为：在 RGB-only、单目标 Cornell 场景中，开放词汇定位能否作为
模块化前端改善抓取检测，以及在共同 crop 和共同 fold 下，几何、单头 CNN
和多头 CNN 如何形成可审计的受控比较。

## 引用与结论边界

- 只使用 `uog_dissertation_outline/l4proj.bib` 中已经核对的文献。
- 正文采用“作者（年份）”或“（作者，年份）”形式。
- 不复制外部论文或代码的长段文字；全部内容为项目独立综述和转述。
- 不把 image-wise 结果描述为 object-wise 或未见物体泛化。
- 不把 RGB VLM crop 与 RGB-D、密集输出或物理抓取结果描述为直接公平比较。
- 不把 Cornell rectangle success 描述为真实机器人成功抓取。
- 只引用已经通过 JSON/CSV 审计的本项目结果。

## DOCX 结构

文档包含：

1. 标题和用途说明；
2. “第一章 引言”及五个二级小节；
3. “第二章 文献综述”及六个二级小节；
4. “引用文献提示”列表，列出正文使用的作者—年份和对应项目 BibTeX key。

正文使用中文字体兼容的 Word 样式，段落间距统一，不使用复杂页眉页脚或学校
模板，以便专注内容批注。

## 验收标准

- DOCX 可以被标准 ZIP/DOCX 解析器正常打开。
- 所有预定一级、二级标题均存在且顺序正确。
- 中文正文约 3000–4000 字，不含 `TODO` 或占位符。
- 三个研究问题与实际 Methodology、Results、Discussion 对应。
- 正文使用的引用均能在 `l4proj.bib` 中找到。
- 关于固定目录、五随机种子、image-wise 五折和 object-wise 阻塞的表述互不
  混淆。
- 不写入新的未经验证实验数字或新的性能主张。
