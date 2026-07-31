# 导师会议纪要：视觉算法与项目深化方向

日期：2026-07-23

原始记录：`docs/视觉算法方案研讨.docx`
主题：三种抓取流程、CNN 设计依据、实验可信度与 MSc 项目深化

## 文档说明

原始 Word 文档由英文语音转录和自动翻译组成，存在大量术语误识别。本纪要依据机器人视觉、抓取检测和上下文进行了修订。

可以较有把握修正的术语包括：

| 原始识别 | 应为 |
|---|---|
| VM / virtual machine | VLM |
| grounding dinosaur / 接地恐龙 | Grounding DINO |
| American cable news network | CNN |
| counter analysis | contour analysis |
| grass / grass point | grasp / grasp point |
| unit | U-Net |
| pose estimator / post estimator | pose estimator |
| group / grouper | grasp / gripper |
| multi, head network | multi-head network |
| loss shown as an image | Grad-CAM、attention map 或类似可解释性方法 |
| most angle for the buck | most bang for the buck |
| PyTorch virtual environment | 很可能是 PyBullet；也可能同时讨论了 Gazebo 等环境 |

无法从音频文本可靠恢复的个别句子没有强行补全；相关位置使用“推测”或“不确定”标记。

---

# 一、导师意见摘要

## 1. 当前成果可以作为 pilot，但技术基础偏简单

导师认可当前三条流程已经产生可用结果，但认为：

- 传统阈值、轮廓和基础 CNN 都属于较成熟甚至较旧的方法；
- 仅证明基础 CNN 比简单几何规则略好，不足以形成很强的 MSc 技术贡献；
- 需要说明 CNN 结构由什么文献、设计原则或失败分析指导；
- 应考虑更现代的网络、可解释性分析或多头结构。

导师的核心判断可以概括为：

> 当前工作“looks fine”并且已经形成 pilot result，但若要达到更有说服力的 MSc 水平，需要再向前推进一步。

## 2. 需要展示 CNN 结构及设计依据

导师明确关注：

- CNN 是否只是最基础的网络；
- 为什么选择当前层数和结构；
- 是否参考了已有抓取网络；
- 为什么没有考虑更现代的多头网络；
- 是否可以展示模型关注图像的哪个区域。

导师提到的可能方向：

- U-Net 或编码器—解码器结构；
- 多头网络；
- attention visualisation；
- Grad-CAM 或类似可解释性方法；
- 分别预测抓取位置、角度或其他变换参数的网络头。

## 3. 结果必须与现代文献比较

导师指出，56%–75% 的结果不能只在三个自建流程之间比较，还要回答：

> 其他抓取检测论文在相同或相近任务上报告了什么结果？

导师接受“本研究的目标是比较特定架构配置，而不是击败 SOTA”这一论证，但要求：

- 明确承认现代 SOTA 方法可能显著更强；
- 把当前方法定位为受控实验或轻量基线；
- 在论文中提供相关文献结果；
- 解释数据划分、指标和输入模态不同，避免不公平地直接比较数字。

## 4. 未见物体测试集的可比性需要重新检查

导师对“未见物体成功率高于全量结果”表示怀疑，认为可能存在：

- 测试物体比训练物体更简单；
- 训练集和测试集物体类别或几何复杂度不可比；
- 因此82.35%的结果可能被测试集构成放大。

导师要求不要简单把该结果解释为强泛化，应检查：

- 训练、验证和测试目录分别包含哪些物体；
- 各集合是否具有类似的形状和难度；
- 是否属于同类物体或可比较的抓取任务；
- 给出每个集合的物体示例图；
- 最好报告按物体类别或形状复杂度分组的结果。

更安全的结论是：

> CNN在固定的85个测试样本上高于几何后端，但该测试集是否能代表一般未见物体仍需进一步验证。

## 5. 当前只预测抓取框，没有实际验证抓取

导师指出当前系统只估计：

- 抓取位置；
- 抓取方向；
- 抓取矩形。

但没有验证：

- 预测姿态能否真的抓起物体；
- 如何把二维抓取框转换为机械臂或夹爪运动；
- 抓取失败后如何重新估计和恢复。

这限制了项目的机器人系统深度。

## 6. 建议探索小规模虚拟机器人演示

导师建议调查一个简单的 Python 机器人仿真环境。根据上下文，他最可能指的是 **PyBullet**，同时可能也提到 Gazebo 或其他机器人仿真工具。

建议的渐进路线是：

1. 选择一个已有机器人、夹爪、相机和物体的仿真环境；
2. 确认环境中可使用哪些物体；
3. 获取虚拟相机图像；
4. 先让当前系统在仿真图像上输出抓取姿态；
5. 如果可行，再把预测姿态转换为夹爪位置；
6. 最后才尝试实际执行抓取。

导师强调不要一开始就追求完整闭环，避免投入大量时间却没有结果。

## 7. 建议把任务拆成有独立价值的小步骤

导师建议的最小可交付结果包括：

- 在选定的虚拟环境中加载一个合适物体；
- 使用虚拟相机获得 RGB 图像；
- 运行现有定位和抓取姿态估计；
- 在图像或仿真环境中显示预测抓取位置与方向；
- 判断该物体是否可抓；
- 如有余力，再执行一次简单抓取。

这条路线的原则是：

> 每完成一步都应形成可展示的结果，即使最后没有完成完整机器人抓取，项目也不会“一无所有”。

## 8. 完整机器人抓取属于多阶段过程

导师提醒真实抓取通常不是一次性从远处直接移动到最终姿态，而是：

```text
目标识别
→ 粗略抓取姿态估计
→ 移动到预抓取位置
→ 近距离重新估计
→ 调整夹爪姿态
→ 闭合夹爪
→ 抓取后成功/失败判断
→ 必要时恢复或重试
```

导师用航天器对接作为类比：先移动到附近，再不断重新估计位置与方向，最后接触。

## 9. 时间有限，应选择投入产出比最高的方向

导师同时强调：

- 完整自监督、真实机器人、多阶段闭环可能达到博士项目规模；
- 当前只剩几周，不应把项目无限扩大；
- 简单网络仍然可以保留；
- 应选择自己最有把握得到结果的扩展；
- 虚拟机器人连接或现代网络分析可以作为提升 MSc 深度的一步。

---

# 二、建议行动优先级

## 必须完成：论文与实验可信度

### A. 展示并论证 CNN 结构

- 在方法章节加入网络结构图或表格；
- 说明四个卷积模块、GAP和回归头；
- 说明为什么使用轻量网络；
- 区分已有文献依据和自主工程选择；
- 不把基础 CNN 声称为新架构。

### B. 增加现代文献比较

- 整理 Cornell、Jacquard或相近二维抓取论文；
- 报告其输入、数据划分、指标和结果；
- 明确哪些结果不能直接横向比较；
- 将当前方法定位为受控、轻量、模块化实验。

### C. 检查 train/val/test 可比性

- 为目录01–06、07–08、09–10分别制作样例图；
- 列出主要物体类型、细长程度和形状复杂度；
- 检查测试集是否明显更简单；
- 修改“未见物体泛化更好”的措辞；
- 只在同一85样本上比较 CNN 与几何后端。

## 高价值扩展：现代化网络或解释性

选择至少一个范围较小的方向：

1. 使用 Grad-CAM 或特征热图显示 CNN 关注区域；
2. 将单一回归头拆成多头：
   - centre head；
   - size head；
   - orientation head；
3. 与一个预训练轻量骨干比较，例如 ResNet18 或 MobileNet；
4. 增加一个角度专用损失或单位圆约束；
5. 对 CNN 进行逐样本失败分析。

## 可选高影响扩展：虚拟机器人 pilot

建议先做可行性调查，不直接承诺完整抓取：

1. 调查 PyBullet、Gazebo、Webots 或 MuJoCo；
2. 优先选择 Python 接口简单、已有夹爪和相机示例的环境；
3. 找到与 Cornell 物体形状相近的仿真模型；
4. 从仿真相机获取一张图并运行现有系统；
5. 把预测抓取框显示回仿真图像；
6. 若坐标转换可控，再尝试预抓取姿态或一次夹取。

---

# 三、修订后的双语会议对话

以下不是逐字法律式转录，而是删除重复口头语后，根据上下文恢复的技术对话。英文尽量保留原意，中文是重新翻译。

## 1. 三条实验流程

**Student**

> I compare three pipelines. The first is a traditional computer-vision baseline using thresholding and contour analysis on the complete image. The second uses Grounding DINO for localisation, followed by a geometric grasp back end. The third uses the same Grounding DINO front end but replaces the geometric back end with a lightweight CNN.

中文：

> 我比较了三条流程。第一条是在完整图像上使用阈值分割和轮廓分析的传统计算机视觉基线；第二条使用 Grounding DINO 定位，再连接几何抓取后端；第三条使用相同的 Grounding DINO 前端，但将几何后端替换为轻量 CNN。

**Supervisor**

> Are the VLM-plus-geometry and VLM-plus-CNN systems based on standard networks and methods developed by other people?

中文：

> VLM加几何和VLM加CNN这两套系统，是基于其他人已经开发的标准网络和方法吗？

**Student**

> Because the dataset is small, I read relevant papers, extracted suitable design information, and implemented a lightweight network myself.

中文：

> 因为数据集比较小，我阅读了相关论文，从中提取适合的设计信息，然后自己实现了一个轻量网络。

**Supervisor**

> Then you need to explain what guided the structure of these networks. CNNs are very basic. You should show the architecture and explain whether other researchers have used this particular design.

中文：

> 那么你需要解释是什么指导了这些网络的结构设计。CNN本身是非常基础的技术，你应该展示架构，并说明其他研究是否使用过这种具体设计。

## 2. 基础 CNN 不足以形成较强的新意

**Student**

> I wanted to create a simple working system first.

中文：

> 我想先实现一个能够工作的简单系统。

**Supervisor**

> I understand that as a first step, but I strongly recommend that you investigate a more sophisticated network. You may not need to construct one completely from scratch. You could add another head to an existing modern network.

中文：

> 我理解这可以作为第一步，但我强烈建议你研究更复杂的网络。你不一定需要从头构建，可以在已有的现代网络上增加新的输出头。

**Supervisor**

> The base CNN architecture you are discussing is based on ideas that have existed for many years. You need something more contemporary if you want stronger MSc-level technical depth.

中文：

> 你正在讨论的基础CNN架构建立在已经存在很多年的思想上。如果想达到更强的MSc技术深度，需要加入更现代的内容。

## 3. U-Net、多头网络和可解释性

**Supervisor**

> You could investigate a multi-head network. Another possibility is a U-Net-style encoder–decoder. A U-Net compresses the image to a bottleneck and then reconstructs a spatial output.

中文：

> 你可以研究多头网络。另一个可能方向是U-Net式编码器—解码器。U-Net先将图像压缩到瓶颈表示，再恢复为空间输出。

**Supervisor**

> You could also show where the network is looking. There are methods that highlight which image regions contributed to a classification or prediction.

中文：

> 你还可以展示网络在看哪里。有一些方法可以突出显示图像中哪些区域对分类或预测产生了贡献。

修订说明：

导师一度未能想起具体名称。从描述看，他可能指：

- Grad-CAM；
- class activation maps；
- attention maps；
- saliency maps。

**Student**

> Perhaps I should investigate how the CNN attends to the observed image.

中文：

> 或许我应该研究CNN在观察图像时关注哪些区域。

**Supervisor**

> Yes. That would show where the information used for the prediction comes from. You could potentially apply interpretability visualisation to both the VLM and CNN components.

中文：

> 是的，这能展示预测所使用的信息来自图像的什么位置。你可以考虑对VLM和CNN组件都使用可解释性可视化。

## 4. 与现代文献比较

**Student**

> The traditional computer-vision baseline achieves 56.95 per cent. After adding Grounding DINO localisation, the geometric pipeline reaches 73.33 per cent. This is the largest improvement in the project.

中文：

> 传统计算机视觉基线达到56.95%。加入Grounding DINO定位后，几何流程达到73.33%，这是项目中最大的提升。

**Supervisor**

> Yes, for your configuration. But how does that compare with the broader literature? In robotics, much higher values have been reported. You need to compare your results with what other researchers report.

中文：

> 对于你的配置来说是这样。但它与更广泛的文献相比如何？机器人抓取研究中报告过高得多的数值。你需要与其他研究的结果进行比较。

**Student**

> My primary objective is to compare these controlled scenarios. I am aware that state-of-the-art systems perform better; this experiment is intended to show the improvement produced by this particular architecture.

中文：

> 我的主要目标是比较这些受控场景。我知道最先进系统表现更好；本实验是为了展示这一特定架构配置带来的提升。

**Supervisor**

> That is a possible argument. The difficulty is that the baseline systems are very old, so almost any modern approach may perform better. You now need to show whether your approach improves on something more contemporary in grasp-point localisation.

中文：

> 这可以作为一种论证，但问题是你的基线方法非常旧，几乎任何现代方法都可能表现更好。你需要进一步说明该方法相对于更现代的抓取点定位方法是否有改进。

## 5. CNN结果与未见物体测试

**Student**

> The CNN achieves an average success rate of 74.51 per cent across five runs. It also produces the highest mean IoU, suggesting better grasp position and size prediction. It performs better on the unseen-object test set.

中文：

> CNN五次实验的平均成功率为74.51%，并取得最高的平均IoU，说明位置和尺寸预测更好。它在未见物体测试集上也表现更好。

**Supervisor**

> That is surprising unless the unseen objects are simpler. Training and test results must come from datasets or subsets that are genuinely comparable; otherwise the comparison may not mean what you think it means.

中文：

> 这有些令人意外，除非未见物体更加简单。训练和测试结果必须来自真正可比较的数据集或子集，否则比较结果可能不代表你认为的含义。

**Supervisor**

> Show me the training objects and test objects. If the training set contains geometrically complicated objects but the test set contains simple objects, then high unseen-object performance does not demonstrate general generalisation.

中文：

> 展示训练物体和测试物体。如果训练集包含几何结构复杂的物体，而测试集主要是简单物体，那么较高的未见物体结果并不能证明一般意义上的泛化。

**Supervisor**

> You may be comparing apples and oranges. The result would then show performance on an easier unseen subset, not necessarily on a comparable unseen class.

中文：

> 你可能在进行不可比的比较。这样的结果只能说明模型在更简单的未见子集上的表现，而不一定代表对可比新类别的泛化。

## 6. 当前没有执行真实抓取

**Supervisor**

> What you are doing is estimating a position and orientation at which to attempt a grasp, but you do not actually attempt to grasp the object using that estimate.

中文：

> 你现在做的是估计一个抓取位置和方向，但并没有依据该估计真正尝试抓取物体。

**Student**

> I currently generate only the grasp rectangle.

中文：

> 我目前只生成抓取矩形。

**Supervisor**

> Do you think you have enough time to attempt a virtual grasp, or would that be too difficult in the time available?

中文：

> 你认为剩余时间足够尝试一次虚拟抓取吗？还是这在现有时间内过于困难？

## 7. 虚拟机器人环境建议

**Supervisor**

> You could investigate a simple Python physics or robotics environment—most likely PyBullet, or another environment with predefined robots and objects.

中文：

> 你可以调查一个简单的Python物理或机器人环境——很可能是PyBullet，或者其他带有预定义机器人和物体的环境。

**Supervisor**

> First check which objects are available and whether they are comparable with the objects used by your current system. Then collect virtual camera images and run your existing pose estimator on those images.

中文：

> 首先检查环境中有哪些物体，以及它们是否与你当前系统使用的物体具有可比性。然后获取虚拟相机图像，并在这些图像上运行现有姿态估计器。

**Supervisor**

> Even before attempting a pick, showing that the current system can detect an object and produce a grasp pose in the new environment would be a meaningful first step.

中文：

> 即使还没有真正执行抓取，只要能证明当前系统可以在新环境中检测物体并产生抓取姿态，也会是一个有意义的第一步。

**Supervisor**

> If that works, it may then be relatively straightforward to position the gripper using the predicted pose and attempt a simple pick. But the risk is spending too much time and failing to connect the complete system.

中文：

> 如果这一步可行，下一步或许可以使用预测姿态设置夹爪位置，并尝试简单抓取。但风险是花费大量时间，最终仍无法连接完整系统。

## 8. 把大任务拆成小步骤

**Supervisor**

> The large goal is a virtual grasp. If that is too ambitious, begin with a smaller goal: make the system produce an initial grasp estimate in the virtual environment.

中文：

> 大目标是完成虚拟抓取。如果目标过大，就从更小的目标开始：让系统先在虚拟环境中产生初始抓取估计。

**Supervisor**

> The first practical task is to identify the environment, select appropriate objects, obtain images, and determine whether your existing system works on them.

中文：

> 第一个实际任务是确定仿真环境、选择合适物体、获取图像，并确认现有系统能否在这些图像上工作。

**Supervisor**

> If you use new objects, repeat your image-processing and evaluation procedure. If you can use the same or similar objects, you already have evidence that the pose estimator can operate on them.

中文：

> 如果使用新物体，需要重复图像处理和评估流程。如果可以使用相同或相似物体，你已经有证据表明姿态估计器可以在这些物体上工作。

## 9. 真实抓取是多阶段闭环

**Supervisor**

> A real system does not simply move directly from the initial camera pose to the final grasp. It first detects the object and estimates a rough grasp pose, moves the gripper close to the object, then re-estimates the relative position and orientation.

中文：

> 真实系统不会从初始相机位置直接移动到最终抓取姿态。它会先检测目标并估计粗略抓取姿态，把夹爪移动到物体附近，再重新估计相对位置和方向。

**Supervisor**

> It is similar to docking two spacecraft: first move close, then repeatedly refine orientation and distance until contact.

中文：

> 这类似两艘航天器对接：先移动到附近，然后不断细化方向和距离，直到接触。

**Supervisor**

> A complete pipeline may contain a pre-grasp stage, an approach, a grasp attempt, a post-grasp evaluation, and a recovery or retry decision.

中文：

> 完整流程可能包括预抓取、接近目标、尝试抓取、抓取后评估，以及恢复或重试决策。

## 10. MSc深度与范围控制

**Supervisor**

> A complete self-supervised system with virtual demonstrations, real-world transfer and closed-loop grasping could become a PhD-scale project.

中文：

> 一个包括虚拟示范、自监督学习、真实环境迁移和闭环抓取的完整系统，可能达到博士项目规模。

**Supervisor**

> You have only a few weeks, so select the direction in which you are most confident of obtaining a result. I do not want to make the project unnecessarily complicated.

中文：

> 你只剩几周，因此应选择最有把握得到结果的方向。我不希望把项目变得不必要地复杂。

**Supervisor**

> What you have done so far looks fine as a pilot. However, students often scope projects at undergraduate level. To bring this to the expected MSc level, add something more modern or demonstrate a meaningful connection to a virtual robot.

中文：

> 你目前完成的内容作为pilot是可以的。但学生往往会把项目限定在本科项目的深度。若要达到预期的MSc水平，需要加入更现代的内容，或者展示与虚拟机器人之间有意义的连接。

**Supervisor**

> Do this incrementally so that you do not end up with nothing. A reasonable first question is simply: is the object in the virtual environment graspable, and what transformation would be required to grasp it?

中文：

> 应采用渐进方式，避免最终没有任何结果。一个合理的第一步问题是：虚拟环境中的物体是否可抓，以及需要怎样的变换才能完成抓取？

---

# 四、建议向导师确认的两个术语

原始转录在仿真环境名称处严重失真。下次会议建议直接询问：

> Did you mean PyBullet when you suggested a simple Python-based virtual robotics environment?

中文：

> 您建议的简单Python机器人仿真环境是指PyBullet吗？

导师提到“显示网络在看哪里”的方法时，也可以确认：

> Were you referring to Grad-CAM or another attention-visualisation method when you suggested showing which image regions guide the CNN prediction?

中文：

> 您建议显示哪些图像区域引导CNN预测时，是指Grad-CAM或其他注意力可视化方法吗？

---

# 五、下一次汇报建议展示的内容

1. 当前 CNN 的结构图、参数量和设计依据；
2. 一页现代抓取网络相关文献对比；
3. train/val/test物体示例拼图；
4. 对82.35%未见物体结果的谨慎解释；
5. 一个现代化小扩展：
   - Grad-CAM；
   - 多头回归；
   - 预训练轻量骨干；
6. PyBullet或其他虚拟环境的可行性调查；
7. 如果可行，展示一张虚拟相机图像上的定位框和抓取姿态。
