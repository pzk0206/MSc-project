# 中文毕业论文结构性改写设计

日期：2026-07-26

## 目标

按照 `docs/agent/DISSERTATION_WRITING_GUIDE.md` 的章节字数和内容要求，将
当前英文论文初稿结构性改写为中文。正式覆盖：

```text
uog_dissertation_outline/l4proj.tex
```

改写不是逐句翻译。现有已经审计的实验数据、图表、引用、复现路径和结论边界
必须保留，但论证结构、章节篇幅、研究问题对应关系和批判性分析需按指南补强。
英文版本保留在 Git 历史提交 `a4d62d6`，不另建双语副本。

本设计取代此前只生成 Introduction/Literature Review 中文 DOCX 的范围；不再
执行 `docs/superpowers/plans/2026-07-26-chinese-intro-literature-docx.md`。

## 语言与排版

- 标题、摘要、章节、小节、正文、图表标题和附录说明改为中文。
- 专有名词首次出现时使用“中文（英文）”，后续可使用通用缩写。
- 作者姓名、论文标题、期刊/会议名称、BibTeX 数据和引用 key 保持英文。
- 数学公式、代码命令、文件路径、模型名称和指标缩写保持原格式。
- 在 `l4proj.tex` 中加载 `xeCJK`，使用当前系统已有字体：
  - 中文正文：`Noto Serif CJK SC`
  - 中文无衬线：`Noto Sans CJK SC`
  - 中文等宽：`Noto Sans Mono CJK SC`
- 继续使用 Tectonic/XeTeX 编译，不修改学校模板的主要版式。

## 章节设计

### Abstract 摘要（300–400 中文字）

摘要最后撰写，使用一个紧凑段落覆盖：

1. 机器人抓取感知与开放词汇定位的背景；
2. 核心研究问题；
3. Cornell RGB 数据、传统 CV、Grounding DINO、几何/CNN 后端和
   image-wise 五折方法；
4. 最重要的已验证数字；
5. 模块化定位的意义；
6. RGB-only、image-wise 和离线矩形评价限制。

摘要不引用文献，不引入正文之外的新结论。

### Acknowledgements 致谢

使用简短、客观的两句文字：

- 感谢导师 Dr Jan Paul Siebert 的指导与反馈；
- 感谢 University of Glasgow School of Computing Science 提供学习和研究
  环境。

不虚构家庭、同学、资助或个人经历。

### Introduction 引言（约 1000 中文字）

保留五个小节：

1. 背景与动机；
2. 研究问题；
3. 研究目标与具体任务；
4. 项目范围；
5. 论文结构。

三个研究问题固定为：

1. 在 Cornell RGB 场景中，Grounding DINO 开放词汇定位能否改善传统整图
   几何抓取流程？
2. 在共同 VLM crop 下，几何后端、单头 CNN 和多头 CNN 分别表现出什么
   优势与局限？
3. 在相同 image-wise manifest 下，多头回归是否优于单头回归？

引言解释研究价值，但不展开具体实验数字或重复文献综述。

### Literature Review 文献综述（3000–4000 中文字）

按主题组织而非逐篇罗列，保留六个小节：

1. 机器人抓取检测；
2. 二维抓取矩形与 Cornell 基准；
3. 传统计算机视觉与深度学习；
4. 视觉语言模型与开放词汇定位；
5. 轻量和参数高效适配；
6. 研究缺口。

五个重点分析对象：

1. **Lenz et al. (2015)**：两阶段候选评分、RGB-D、image-wise/object-wise
   协议；优点是系统化 Cornell 深度学习评估，限制是候选搜索成本和输入/划分
   与本项目不同。
2. **Redmon and Angelova (2015)**：全局直接回归与实时性；优点是端到端速度，
   限制是单全局输出表达力和 RGB-D/标准五折差异。
3. **Morrison et al. (2018), GG-CNN**：深度密集抓取图、闭环控制和
   `sin(2θ), cos(2θ)`；优点是多抓取和实时反馈，限制是深度输入、物理系统
   变量和与 RGB 单矩形任务不可直接比较。
4. **Liu et al. (2023), Grounding DINO**：文本条件开放集检测；优点是无需
   固定类别表，限制是只提供语义定位框，不直接建模抓取可行性。
5. **Vuong et al. (2024), Language-Driven Grasp Detection**：语言指定目标的
   联合抓取任务；优点是复杂语义条件，限制是数据、任务和指标与 Cornell
   单目标场景不同。

辅助文献用于建立发展脉络：

- Jiang et al. (2011)：二维矩形表示；
- Kumra and Kanan (2017)：RGB-D ResNet；
- Kumra et al. (2020)：GR-ConvNet；
- Li et al. (2022)：Gaussian-guided 密集预测；
- Radford et al. (2021)：CLIP 视觉语言对齐；
- Hu et al. (2022) 与 Dettmers et al. (2023)：LoRA/QLoRA。

批判性比较必须覆盖：

- RGB 与 RGB-D；
- 固定目录、image-wise 和 object-wise；
- 单矩形与密集多抓取；
- 离线 rectangle metric 与物理抓取成功率；
- 数据增强、模型容量和具体 fold 成员对结果可比性的影响；
- 定位覆盖率不等于定位准确率或抓取成功。

Literature Gap 固定为：现有工作分别研究高性能抓取预测和开放词汇/语言感知，
但较少在 RGB-only、单目标 Cornell 场景中把开放词汇定位作为可替换前端，并在
共同 crop、共同评价实现和共同 fold 下，对几何、单头 CNN 和多头 CNN 后端做
可审计的受控比较。

### Methodology 方法论（1000–2000 中文字）

将现有英文方法完整改写为中文，并加强“为什么选择该方法”：

- 解释使用 Cornell、RGB-only 和单目标 prompt 的控制变量理由；
- 解释三条 pipeline 如何分别隔离定位增益与后端差异；
- 解释固定目录结果、五随机种子和 image-wise 五折回答不同问题；
- 解释为什么不生成 object-wise 结果；
- 解释选择 Cornell rectangle metric 的优点与限制；
- 说明严格确定性、共同 manifest、结果哈希和测试集隔离如何降低实验风险。

增加 Research Ethics 小节：

- 本项目没有人类参与者、私人数据或问卷；
- 使用公开数据、公开论文和预训练模型；
- 外部代码或方法必须标注来源；
- 不夸大 image-wise 泛化、定位覆盖率或物理执行含义；
- 保存产物、环境和哈希以支持复核；
- 计算资源和能源使用保持在完成研究问题所需的范围。

### Findings / Results 结果（3000–4000 中文字）

保留并中文化现有三张核心表和证据图：

- 完整 885 样本方法比较；
- 固定 85 样本比较；
- image-wise 五折单头/多头比较；
- Cornell split contact sheet；
- 后端失败案例图；
- 附录固定 split 表。

结果按研究问题组织：

1. Grounding DINO 前端带来的定位/几何改进；
2. 几何与 CNN 在 IoU、角度和逐样本结果上的互补性；
3. 单头与多头在确定性五 seed 和 image-wise 五折中的比较。

所有数字必须继续来自现有审计产物。主要正式数字包括：

- 传统 CV：56.95%，IoU 0.3360，角度 29.62°；
- VLM + geometry：73.33%，IoU 0.4182，角度 14.81°；
- 确定性五 seed 单头：74.55% ± 1.77%；
- 确定性五 seed 多头：75.59% ± 1.90%；
- image-wise 单头：635/885，71.75%，IoU 0.4390，角度 17.74°；
- image-wise 多头：647/885，73.11%，IoU 0.4580，角度 17.40°。

结果章节描述观察和数据，不把推测原因写成已验证因果结论。

### General Discussion 综合讨论

中文重写现有 Discussion，避免重复 Results。重点解释：

- 定位前端是最大增益来源；
- CNN 主要改善位置和尺寸，几何角度先验仍有价值；
- 多头相对单头的提升真实但幅度有限；
- 固定测试子集可能更容易；
- image-wise 不证明未见物体泛化；
- 与 RGB-D 密集方法和物理实验仅有限可比；
- 模块化 VLM 前端的实践意义与理论含义；
- 失败案例只能支持观察到的错误类型，不能自动确认因果。

### Conclusion 结论（约 1000 中文字）

使用四个小节：

1. 研究总结；
2. 研究问题回答；
3. 贡献与实践意义；
4. 限制与建议。

结论必须逐条回答三个研究问题，并把 future work 限定为：

- 获取权威 object-instance 映射后进行 object-wise 五折；
- 深度融合和密集多抓取输出；
- 更强的方向约束或几何—学习混合后端；
- 真实机器人闭环验证；
- 更复杂的开放词汇场景和提示词敏感性实验。

### Appendices 附录

中文化说明文字，保留命令、路径、环境版本和表格数据。引用条目保持英文。

## 引用与外部来源边界

- 所有正式引用来自 `uog_dissertation_outline/l4proj.bib`。
- 重点论文的技术描述必须由论文原文、作者页面、DOI 或官方开放访问页面支持。
- 不使用二手博客支持技术结论。
- 不复制论文长段原文；全部使用独立中文概括和批判性分析。
- 代码来源遵循 `AGENTS.md` 的标注规则。
- Park et al. (2018) 撤回稿不进入正式性能排名。

## 保留与修改范围

必须保留：

- LaTeX 表格、图像文件和 label/ref 关系；
- BibTeX key；
- 已审计的实验数字；
- 复现命令和输出路径；
- 固定目录、五随机种子、image-wise 五折之间的统计口径；
- object-wise 元数据阻塞结论。

允许修改：

- 章节和小节标题；
- 正文组织和措辞；
- 表格与图像 caption；
- 研究问题、研究目标、贡献和章节导航；
- Methodology 的选择理由、Ethics 和限制；
- Results 与 Discussion 的边界。

## 验证与验收

### 内容检查

- `l4proj.tex` 除教育复用表单注释外不含 `\todo{}`。
- 章节字数/中文字符数接近指南目标，不用重复内容机械凑字数。
- 三个研究问题在 Introduction、Results、Discussion 和 Conclusion 中一致。
- 五篇重点文献均有实质性批判分析，不是摘要式罗列。
- Literature Gap 与研究问题和实验设计一致。
- Ethics 符合无参与者、公开数据的实际情况。

### 数字检查

- 从正式 JSON/CSV 重新读取数字，与 LaTeX 表格和正文逐项核对。
- Pooled 指标、fold 标准差和五 seed 标准差不混写。
- 单头和多头不调换。
- 固定目录不标成标准 object-wise。
- Image-wise 不标成未见物体泛化。

### 引用检查

- 每个 `\citep`/`\citet` key 都存在于 `l4proj.bib`。
- 不新增无来源的论文性能数字。
- 重点文献的输入、输出、数据划分和指标描述与原始来源一致。

### 构建与视觉检查

- Tectonic 编译退出码为 0，生成中文 PDF。
- PDF 能提取中文文字，不出现大面积缺字方框。
- 目录、章节、表格、公式、图像和引用可见。
- 对 Abstract、Literature Review、Results 和 Conclusion 页进行截图或文本
  回读检查。
- 既有测试继续通过，文档改写不影响实验代码。

## 完成定义

当且仅当以下条件全部满足时，本轮完成：

1. 整篇现有英文初稿已被结构性中文稿覆盖；
2. 所有核心章节满足指南内容要求；
3. 全部 `TODO` 仅允许保留用户明确要求自行填写的教育复用个人信息注释；
4. 数字、引用和结论边界通过独立检查；
5. 中文 PDF 成功生成并可阅读；
6. `CURRENT_STATUS.md` 和 `WORKLOG.md` 已记录论文中文改写状态；
7. 英文旧版本可从 Git 历史恢复。
