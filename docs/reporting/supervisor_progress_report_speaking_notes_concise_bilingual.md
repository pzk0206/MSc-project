# Concise Supervisor Presentation Script

对应 PPT：`supervisor_progress_report_2026-07-23_en.pdf`  
预计时长：3–4 分钟

## Opening

> Good morning. Today I will briefly introduce the current progress of my MSc project on VLM-guided 2D robotic grasp detection.
>
> I will present the project design, the main experimental results, and my next steps.

中文提示：

- 简单说明研究主题和汇报结构。
- 不需要在开场解释技术细节。

---

# Slide 1 — Project Overview

> The main research question is whether a pretrained open-vocabulary vision-language model can improve 2D grasp detection by providing object localisation.
>
> I use the Cornell Grasping Dataset, with 885 samples. A grasp prediction is successful if its IoU is at least 0.25 and its angle error is no more than 30 degrees. These two conditions must be satisfied at the same time.

中文提示：

- VLM 负责定位目标区域，不直接完成抓取。
- 成功标准：IoU ≥ 0.25，同时角度误差 ≤ 30°。

> I compare three pipelines.
>
> The first is a traditional computer-vision baseline using thresholding and contour analysis on the complete image.
>
> The second uses Grounding DINO for localisation, followed by a geometric grasp back end.
>
> The third uses the same Grounding DINO front end, but replaces the geometric back end with a lightweight CNN.

中文提示：

- 按 PPT 从左到右介绍三条流程。
- 强调后两种方法使用相同的 VLM 定位前端。

> The CNN receives a 224-by-224 RGB crop and predicts the grasp centre, width, height, and orientation.

中文提示：

- CNN 结构不用在主讲中展开。
- 如果老师追问，再解释四个卷积模块和六个输出。

### Transition

> The next slide shows the main experimental results.

---

# Slide 2 — Experimental Results

> The traditional computer-vision baseline achieves a success rate of 56.95 per cent.
>
> After adding Grounding DINO localisation, the geometric pipeline reaches 73.33 per cent. This is the largest improvement in the project, showing that object localisation is the main source of performance gain.

中文提示：

- 最重要结果：56.95% 提升到 73.33%。
- VLM 定位贡献最大。

> The CNN achieves an average success rate of 74.51 per cent across five runs. It also produces the highest mean IoU, which suggests better prediction of grasp position and size.
>
> However, the geometric method has the lowest angle error, at 14.81 degrees.

中文提示：

- CNN：位置和尺寸更好。
- 几何后端：角度更准确。
- 不要夸大 CNN 的成功率提升，因为74.51%与73.33%比较接近。

> On the unseen-object test set, the CNN reaches 82.35 per cent, compared with 75.3 per cent for the geometric method. This suggests that the CNN has better generalisation to unseen objects.

中文提示：

- 使用 `suggests`，不要使用 `proves`。
- 测试集包含85个样本。

> The failure analysis also supports this result. Among the 236 failures of the geometric pipeline, 126 already have an acceptable angle, but fail mainly because the position or size is inaccurate.

中文提示：

- 失败分析说明主要问题是 IoU，而不是只有角度问题。
- 这也是引入 CNN 的原因。

> Overall, the VLM provides the largest improvement, the CNN is better for position and size, and the geometric method is better for orientation.

中文提示：

- 记住三个对应关系：
  - VLM：localisation
  - CNN：position and size
  - Geometry：orientation

### Transition

> Based on these results, the main experimental stage is now complete.

---

# Slide 3 — Current Status and Next Steps

> I have now completed the three experimental pipelines, the full-dataset evaluation, five CNN runs, the unseen-object evaluation, and the failure analysis.

中文提示：

- 实验部分基本完成。
- 论文写作还没有完成。

> The current study has three main limitations.
>
> First, it is evaluated only on the Cornell dataset. Second, it is an offline 2D perception study without physical robot experiments. Third, 885 out of 885 means that Grounding DINO returned a box for every sample, but it does not mean that every box is perfectly accurate.

中文提示：

- 三个局限：
  1. 只有 Cornell；
  2. 没有真实机器人；
  3. 检测覆盖不等于定位完全准确。

> My next priority is to complete the dissertation and organise the experimental figures and results.
>
> If time permits, I will perform a more detailed CNN error analysis and explore a hybrid back end that uses the CNN for position and size and the geometric method for orientation.

中文提示：

- 论文写作是当前重点。
- 混合后端属于可选扩展，不要承诺一定完成。

---

# Closing

> To conclude, Grounding DINO localisation significantly improves the traditional grasp-detection pipeline.
>
> The CNN provides better position, size, and unseen-object performance, while the geometric method provides a useful orientation prior.
>
> Thank you. I am happy to answer any questions.

中文提示：

- 最后再总结两种后端的互补性。
- 讲完停顿，等待老师提问。

---

# One-Minute Version

如果老师只给一分钟，可以直接使用下面这段：

> This project evaluates whether Grounding DINO can improve 2D robotic grasp detection on the Cornell dataset.
>
> I compare traditional computer vision, VLM-guided geometric grasping, and VLM-guided CNN regression. The traditional baseline achieves 56.95 per cent, while adding VLM localisation increases the result to 73.33 per cent. The CNN achieves 74.51 per cent across five runs and performs better on the unseen-object test set.
>
> The main conclusion is that VLM localisation provides the largest improvement. The CNN is better for position and size, while the geometric method is better for orientation. The experiments are complete, and my next priority is dissertation writing.

