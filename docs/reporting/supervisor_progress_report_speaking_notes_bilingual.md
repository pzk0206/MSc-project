# Supervisor Progress Report — Bilingual Speaking Notes

汇报人：Pang Zhenkun  
项目：Evaluating Open-Vocabulary Vision-Language Models for 2D Robotic Grasp Detection  
对应文件：`supervisor_progress_report_2026-07-23_en.pdf`

## 使用方法

- 正常语速下，主讲部分约为 6–8 分钟。
- 英文正文可以直接讲；“中文提示”用于理解逻辑，不需要逐句翻译给老师。
- 不需要把每个数字都念出来，优先强调成功率、未见物体结果和主要结论。
- 如果时间较短，可以只讲每页的“简短版本”。

---

## Opening — 开场

### 推荐版本

> Good morning. Today I would like to give a brief update on my MSc project, which investigates whether an open-vocabulary vision-language model can improve two-dimensional robotic grasp detection.
>
> I will first introduce the research question and the three experimental pipelines. I will then present the main results, and finally explain the current limitations and my next steps.

中文提示：

- 开场先说明这是 MSc 项目进展汇报。
- `investigates whether` 表示“研究是否”，比直接说 `proves` 更严谨。
- 告诉老师汇报分为三部分：方法、结果、下一步。

### 20 秒简短版本

> Today I will briefly present the progress of my MSc project on VLM-guided 2D robotic grasp detection. I have completed three experimental pipelines, evaluated them on the Cornell Grasping Dataset, and identified the main strengths and limitations of each method.

中文提示：

- 如果老师时间紧，可以直接用这一段。
- 重点突出：三条流程已经完成，并且已经得到可比较的实验结果。

---

# Page 1 — Project Overview

## 1. Research question

> The main research question is whether a pretrained open-vocabulary vision-language model can serve as an object localisation front end and improve 2D grasp rectangle detection.

中文提示：

- `pretrained`：预训练的。
- `open-vocabulary`：开放词汇，表示模型可以根据文本提示定位不同类别的目标。
- `localisation front end`：定位前端。
- 不要说 VLM 直接完成了抓取。它主要负责找到物体，抓取框由后端产生。

> The motivation is that conventional image-processing methods often operate on the complete image and can be affected by the background. My approach first uses Grounding DINO to identify the target region. The grasp back end then only needs to analyse this smaller and more relevant region.

中文提示：

- 传统方法容易受背景、桌面颜色和光照影响。
- VLM 的作用是缩小搜索范围。
- `relevant region` 表示“与任务相关的区域”。

> To isolate the effect of the grasp back end, I keep the VLM localisation stage fixed and compare two different back ends: an explicit geometric method and a learned CNN regressor.

中文提示：

- 这是实验设计的重要逻辑：定位前端保持一致，才能比较几何后端和 CNN 后端。
- `isolate the effect` 表示“单独考察某个因素的影响”。

## 2. Dataset and evaluation

> I use the Cornell Grasping Dataset, with 885 usable RGB samples in the current experiment. Grounding DINO returns a localisation box for all 885 samples under the current prompt and threshold settings.

中文提示：

- 一定说 `returns a localisation box`，不要直接说 `100 per cent localisation accuracy`。
- 885/885 表示全部样本都有检测输出，不代表每个定位框都完美。

> A predicted grasp is considered successful if it matches at least one positive ground-truth grasp rectangle and satisfies two conditions at the same time: the intersection over union must be at least 0.25, and the orientation error must be no more than 30 degrees.

中文提示：

- 关键词是 `at the same time`，两个条件必须同时满足。
- 一张 Cornell 图像可能有多个正抓取标注；只要匹配其中一个就算成功。
- IoU 是预测矩形与真实矩形的交集面积除以并集面积。

> This is the standard Cornell rectangle metric used in earlier grasp-detection research, including Jiang and colleagues in 2011 and Lenz and colleagues in 2015.

中文提示：

- 这句话用于说明阈值不是随意选择的。
- 如果老师追问，Lenz 等人的论文明确使用了 30 度方向阈值和 25% IoU 阈值。

## 3. Three experimental pipelines

### Pipeline 1: Traditional CV

> The first pipeline is a traditional computer-vision baseline. It applies colour and brightness thresholding to the complete RGB image, extracts the main contour, and converts it into a minimum-area rotated rectangle.

中文提示：

- 这是最低成本、可解释的基线。
- 它不使用 Grounding DINO，也不使用学习式抓取模型。

### Pipeline 2: VLM plus geometry

> The second pipeline uses Grounding DINO to localise the object. The same type of segmentation and contour analysis is then applied only inside the VLM region. A geometric rule converts the object orientation into a grasp orientation.

中文提示：

- 与传统基线的主要区别是先用 VLM 限定目标区域。
- 几何后端根据物体轮廓和长轴方向产生抓取框。

### Pipeline 3: VLM plus CNN

> The third pipeline uses the same Grounding DINO front end, but replaces the handcrafted geometric back end with a lightweight CNN. The CNN receives a 224-by-224 RGB crop and directly regresses six grasp parameters.

中文提示：

- `handcrafted` 指人工设计的规则。
- CNN 输出中心、尺寸和双角度编码。
- 强调 Grounding DINO 前端保持不变，因此结果差异主要来自抓取后端。

## Page 1 transition

> After implementing these three pipelines, I evaluated them using the same Cornell rectangle metric. The next slide summarises the main results.

中文提示：

- 这是从方法页过渡到结果页的连接句。

## Page 1 简短版本

> The project compares three pipelines: traditional computer vision, VLM-guided geometric grasping, and VLM-guided CNN grasp regression. All methods are evaluated on 885 Cornell samples using the standard rectangle metric: IoU of at least 0.25 and an angle error of no more than 30 degrees.

---

# Page 2 — Experimental Results

## 1. Full-dataset comparison

> On the full Cornell dataset, the traditional computer-vision baseline achieves a success rate of 56.95 per cent.

> After introducing Grounding DINO localisation, the geometric pipeline reaches 73.33 per cent. This is an improvement of approximately 16.4 percentage points, which is the largest performance increase in the study.

中文提示：

- 最重要的结论之一：VLM 定位带来的增益最大。
- 是增加 `16.4 percentage points`，不要说增加 `16.4 per cent`，两者含义不同。

> The five-run CNN result is 74.51 per cent, with a standard deviation of 1.38 percentage points. Its mean best IoU is 0.4510, compared with 0.4182 for the geometric back end.

中文提示：

- 五次实验均值：74.51%。
- 标准差：1.38%，用于说明随机初始化带来的波动。
- CNN 的完整数据集成功率只比几何后端略高，但 IoU 更高。

> This suggests that the CNN is better at estimating the position and size of the grasp rectangle. However, the geometric method achieves the lowest mean angle error, at 14.81 degrees, compared with 16.49 degrees for the CNN.

中文提示：

- CNN：中心位置和尺寸更好。
- 几何后端：角度更准确。
- 两种方法具有互补性。

## 2. Unseen-object test set

> I also evaluate generalisation on the Cornell directories 09 and 10, which contain 85 samples that are not used for CNN training or validation.

中文提示：

- 训练：01–06。
- 验证：07–08。
- 测试：09–10。
- 这样划分是为了减少相似物体泄漏。

> On this unseen-object test set, the geometric back end achieves 75.3 per cent, while the CNN achieves an average of 82.35 per cent across five runs, with a standard deviation of 4.53 percentage points.

> The larger improvement on the unseen-object test set suggests that the CNN has learned visual patterns that generalise better than the fixed geometric rule.

中文提示：

- 使用 `suggests`，不要说 `proves`。
- 测试集只有85个样本，因此标准差较大。
- 结果说明 CNN 的泛化表现更好，但仍需要更多数据集验证。

## 3. Failure analysis

> Before designing the CNN back end, I analysed the 236 failures of the VLM-guided geometric pipeline.

> Among these failures, 126 already had an acceptable orientation, but failed mainly because the predicted centre or size produced insufficient overlap with the ground truth.

中文提示：

- 126/236 约为53.4%。
- 这些样本角度基本正确，主要是 IoU 不达标。
- 这为引入 CNN 回归位置和尺寸提供了实验动机。

> Therefore, the failure analysis indicated that further improvements should focus on grasp position and size, rather than only modifying the orientation rule.

## 4. Main interpretation

> Overall, the VLM localisation front end provides the largest improvement. The CNN back end gives better overlap and unseen-object generalisation, while the geometric back end remains slightly better for orientation estimation.

> These results indicate that the two back ends are complementary. A possible future direction is to use the CNN for position and size, while retaining an explicit geometric prior for the grasp angle.

中文提示：

- 这是结果页最重要的总结。
- 可以用三个词记忆：
  - VLM：localisation；
  - CNN：position and size；
  - Geometry：orientation。

## Page 2 transition

> Based on these results, the main experimental stage is now complete. The final slide presents the current project status, limitations, and next steps.

## Page 2 简短版本

> The VLM provides the largest gain, improving the success rate from 56.95 to 73.33 per cent. The CNN achieves the highest mean IoU and better unseen-object performance, while the geometric method has the lowest angle error. This shows that the CNN and geometric back ends have complementary strengths.

---

# Page 3 — Current Status and Next Steps

## 1. Completed work

> At this stage, all three experimental pipelines have been implemented and evaluated.

> I have completed the full-dataset comparison, five independent CNN runs, the unseen-object evaluation, and the failure-case analysis.

中文提示：

- 三条实验流程已经完成。
- 不要说整个毕业论文已经完成；实验部分完成，论文写作仍在进行。

## 2. Current limitations

> The first limitation is that the current study is evaluated only on the Cornell Grasping Dataset. Therefore, the results cannot yet be assumed to generalise to more complex datasets or real environments.

> Second, this is an offline 2D perception study. It predicts grasp rectangles from stored images, but it does not yet control a physical robot or evaluate closed-loop grasp execution.

> Third, the 885-out-of-885 Grounding DINO result means that the model produced a box for every sample. It should not be interpreted as perfect localisation accuracy, because variations in box quality can still affect the grasp back end.

中文提示：

- 主动承认局限会让汇报更可信。
- 三个局限：单数据集、没有真实机器人、定位覆盖不等于定位准确率。

## 3. Next steps

> My immediate priority is dissertation writing. I will complete the introduction, background, methodology, results, and discussion chapters, and organise the experimental tables and qualitative examples.

> If time permits, I will add a more detailed per-sample analysis of the CNN errors and investigate a hybrid back end that combines CNN position regression with the geometric orientation prior.

> A further extension would be to evaluate the method on a more difficult dataset, such as Jacquard, or eventually on a physical robotic platform.

中文提示：

- `immediate priority`：当前最高优先级。
- Jacquard 和真实机器人属于扩展，不要承诺一定完成。
- 使用 `if time permits` 表示“时间允许的情况下”。

## 4. Closing statement

> To conclude, the current results show that open-vocabulary localisation can substantially improve a conventional grasp-detection pipeline by reducing background interference and focusing the grasp back end on the relevant object region.

> The remaining challenge is no longer simply finding the object. It is producing a grasp rectangle with sufficiently accurate position, size, and orientation.

> Thank you. I am happy to answer any questions.

中文提示：

- 结论不要说“VLM解决了抓取问题”。
- 更准确的说法是：VLM 显著改善目标区域定位，当前瓶颈转移到抓取后端。

## Page 3 简短版本

> The three experimental pipelines and the main evaluation are complete. The current limitations are the Cornell-only evaluation, the lack of physical robot experiments, and the difference between box coverage and box accuracy. My next priority is dissertation writing, followed by optional CNN error analysis and a hybrid grasp back end.

---

# CNN Architecture — 备用技术讲解

## 1. CNN 的设计动机

> Grounding DINO already identifies the target region, so the CNN does not need to perform full-scene object detection. Its task is simplified to predicting one grasp rectangle within the localised crop.

> Because the dataset contains only 885 samples, I designed a lightweight network rather than using a very large pretrained backbone. This reduces computational cost and the risk of overfitting.

中文提示：

- CNN 不是负责全图目标检测。
- 小数据集对应轻量网络。
- 当前网络约43万可训练参数。

## 2. CNN 的具体结构

> The input is a normalised 224-by-224 RGB crop. The convolutional backbone contains four blocks, with channel dimensions increasing from 32 to 64, 128, and 256.

> As the channel depth increases, max pooling gradually reduces the spatial resolution from 224 to 56, 28, 14, and finally 7 pixels.

> Each block uses convolution, batch normalisation, ReLU activation, and max pooling. The earlier layers learn edges and local textures, while the deeper layers represent more global shape and grasp-related information.

中文提示：

- 第一层是5×5卷积、stride=2，然后池化。
- 后面三层是3×3卷积。
- 通道增加、空间尺寸减少是典型 CNN 设计。

> After the fourth convolutional block, I use global average pooling to convert the 256-by-7-by-7 feature map into a 256-dimensional vector.

> Global average pooling substantially reduces the number of parameters compared with directly flattening the complete feature map, which is particularly useful for a small dataset.

中文提示：

- 如果直接 Flatten，是12544维。
- GAP 后只有256维。
- 可以显著减少全连接层参数和过拟合风险。

> The regression head then reduces the feature dimension from 256 to 128, then to 64, and finally to six output values. Dropout is used in the fully connected layers to provide additional regularisation.

## 3. CNN 输出

> The six outputs are the normalised centre coordinates, normalised width and height, and two orientation components: sine of twice the angle and cosine of twice the angle.

> Although a grasp rectangle has five geometric degrees of freedom, the angle is represented by two values. This handles the 180-degree symmetry of a parallel-jaw gripper and avoids a discontinuity at the angle boundary.

中文提示：

- 六个输出：
  - \(c_x\)
  - \(c_y\)
  - width
  - height
  - \(\sin(2\theta)\)
  - \(\cos(2\theta)\)
- 双角编码参考 GG-CNN 类研究。

> During inference, the angle is recovered using half of the two-argument arctangent of the predicted sine and cosine components.

## 4. CNN 训练

> The model is trained using Smooth L1 loss, which is less sensitive to large regression errors than mean squared error.

> I use the Adam optimiser with an initial learning rate of 0.001, a batch size of 32, weight decay, learning-rate reduction based on validation loss, and early stopping.

> The Cornell directories are divided into training directories 01 to 06, validation directories 07 to 08, and test directories 09 to 10, in order to reduce leakage between similar object samples.

中文提示：

- 这部分只有老师追问时再讲。
- 不需要在主汇报中逐个念超参数。

## 5. CNN 设计来源怎么说

> The complete CNN architecture is not copied directly from a single paper. It is a lightweight regression network designed for this project.

> However, the grasp rectangle formulation and evaluation metric follow established Cornell grasp-detection research, and the double-angle orientation encoding is supported by generative grasp-detection literature such as GG-CNN.

中文提示：

- 不要说整个 CNN 都有某一篇论文作为直接依据。
- 准确区分：
  - 网络层数和通道数：自己的工程设计；
  - 抓取矩形表示和评价标准：已有文献；
  - 双角编码：GG-CNN 等论文依据。

---

# Likely Questions and Suggested Answers

## Q1. Why did you choose Grounding DINO?

> I chose Grounding DINO because it supports open-vocabulary object detection using text prompts. This allows the localisation front end to identify target regions without training a new detector specifically for the Cornell object categories.

中文提示：

- 核心：开放词汇、文本提示、无需专门训练 Cornell 检测器。

## Q2. Does 885 out of 885 mean perfect localisation?

> No. It means that the model returned a detection box for every sample under the current settings. Box coverage and box accuracy are different. An inaccurate or overly large box can still reduce the quality of the final grasp prediction.

中文提示：

- 这道题非常可能被问。
- 一定区分 coverage 和 accuracy。

## Q3. Why use the prompt “small object”?

> I used a generic prompt because the experiment focuses on open-vocabulary localisation rather than manually providing the exact object category for every image. In the current Cornell setting, “small object” produced the most reliable sample coverage among the tested prompt choices.

中文提示：

- 不要说这是理论最优 prompt。
- 说它是当前实验设置下表现稳定的通用提示词。

## Q4. Why is the IoU threshold only 0.25?

> The threshold follows the established Cornell rectangle metric. A ground-truth rectangle represents a region of feasible grasps rather than a unique object bounding box, so grasp-detection research commonly uses an IoU threshold of 25 per cent together with a 30-degree orientation threshold.

中文提示：

- 抓取矩形不是普通目标检测框。
- 同一物体可能存在多个有效抓取姿态。

## Q5. Why does the CNN output only one grasp?

> The current CNN is designed as a simple global regression baseline. It selects one training target and predicts one grasp rectangle per crop. This makes the comparison clear and computationally lightweight, but it does not represent all possible valid grasps.

> A future model could predict a dense grasp map or multiple grasp candidates.

中文提示：

- 承认单抓取输出是简化设计。
- 后续可做多抓取或像素级预测。

## Q6. How do you select one label when Cornell has multiple grasps?

> Among the positive grasp rectangles whose centres lie inside the VLM crop, the current implementation selects the rectangle with the largest area as a deterministic training target.

> This simplifies single-output regression, although it also discards other valid grasps. A more advanced approach could compute the minimum loss over all valid ground-truth rectangles.

中文提示：

- 这是当前设计的局限。
- 不要声称面积最大的一定是物理上最稳定的抓取。

## Q7. Why use sine and cosine of twice the angle?

> A parallel-jaw grasp has 180-degree rotational symmetry, so an angle and the same angle plus 180 degrees represent an equivalent gripper pose.

> Using sine and cosine of twice the angle maps these equivalent poses to the same target and removes the discontinuity at the angle boundary.

中文提示：

- 关键公式：
  \[
  \theta \equiv \theta+\pi
  \]
- 恢复：
  \[
  \theta=\frac{1}{2}\operatorname{atan2}(\sin 2\theta,\cos 2\theta)
  \]

## Q8. Why not use a pretrained ResNet?

> A pretrained ResNet could provide stronger features, but it would also introduce a much larger model and make it harder to determine whether the improvement comes from VLM localisation or from a powerful pretrained grasp back end.

> I first used a lightweight CNN to create a controlled and interpretable comparison. Transfer learning is a reasonable future extension.

中文提示：

- 不要否定 ResNet。
- 当前目标是可控、轻量的基线。

## Q9. Why is the CNN improvement small on the full dataset?

> The geometric back end is already strong after VLM localisation, especially for orientation estimation. Therefore, replacing it with a CNN does not produce a large increase in overall success rate.

> The CNN improvement is clearer in mean IoU and on the unseen-object test set, which indicates better position and size estimation rather than a large improvement in every component.

中文提示：

- 全量成功率：74.51% 对 73.33%，差距较小。
- 不能夸大提升。
- CNN 的价值主要在 IoU 和未见物体泛化。

## Q10. Why is the geometric angle more accurate?

> The geometric method directly uses an explicit object-axis rule, so it has a strong orientation prior. The CNN must learn orientation only from the available training examples, which introduces more variation.

> This result motivates a hybrid approach that combines learned position and size with an explicit geometric orientation prior.

中文提示：

- 几何规则在角度上更稳定。
- 这不是 CNN 完全失败，而是两种方法互补。

## Q11. Why is unseen-object performance higher than the full-dataset result?

> The two numbers are calculated on different sample subsets, so they should not be interpreted as a direct difficulty ranking. The unseen-object subset may contain objects or viewpoints that are relatively favourable for the current model.

> The important comparison is between the CNN and geometric back ends on the same 85 test samples.

中文提示：

- 测试集82.35%高于全量74.51%，不代表未见物体一定更简单或模型越陌生越好。
- 数据子集构成不同。
- 应比较同一测试集上的 CNN 与几何方法。

## Q12. What is the main contribution of the project?

> The main contribution is a controlled comparison showing how an open-vocabulary localisation front end affects a conventional 2D grasp-detection pipeline, and how geometric and learned grasp back ends behave under the same localisation input.

> The experiments show that localisation provides the largest gain, while the two grasp back ends have complementary strengths.

中文提示：

- 不要把贡献说成提出了全新的基础模型。
- 贡献主要是系统设计、受控对比、失败分析和实验结论。

## Q13. What would you improve first?

> My first technical improvement would be to combine the CNN position and size estimates with the geometric orientation prior, because the current results provide direct evidence that these components have complementary strengths.

> I would then evaluate the method on a larger and more difficult dataset to test whether the conclusions generalise beyond Cornell.

## Q14. Can this system control a real robot?

> Not yet. The current project evaluates offline 2D grasp perception. A real robotic system would additionally require camera calibration, depth or 3D pose estimation, coordinate transformation, motion planning, collision checking, and closed-loop execution.

中文提示：

- 明确当前范围。
- 不要把二维抓取框等同于完整机械臂控制。

---

# Useful Presentation Phrases

## 引导老师看页面

> As shown in the table...

中文：如表格所示。

> The key point here is...

中文：这里的关键点是……

> I would like to highlight two findings.

中文：我想重点强调两个发现。

> This result should be interpreted carefully.

中文：这个结果需要谨慎解释。

> These two values are evaluated on different subsets.

中文：这两个数值来自不同的数据子集。

## 不确定时的表达

> Based on the current experiments, my interpretation is that...

中文：根据目前的实验，我的理解是……

> I have not tested that directly, so I would treat it as a possible extension.

中文：我还没有直接验证这一点，因此我会把它作为可能的扩展方向。

> I would need an additional ablation experiment to answer that conclusively.

中文：要得出明确结论，还需要额外的消融实验。

## 请求确认问题

> If I understand the question correctly, you are asking whether...

中文：如果我理解正确，您的问题是……

> Do you mean the localisation stage or the grasp-regression stage?

中文：您指的是定位阶段还是抓取回归阶段？

---

# Vocabulary and Pronunciation Notes

| English | 中文含义 | 发音提示 |
|---|---|---|
| grasp | 抓取 | /grɑːsp/，接近“格拉斯普” |
| localisation | 定位 | 英式拼写，/ˌləʊkəlaɪˈzeɪʃən/ |
| regression | 回归 | /rɪˈgreʃən/ |
| geometric | 几何的 | /ˌdʒiːəˈmetrɪk/ |
| orientation | 方向、角度 | /ˌɔːriənˈteɪʃən/ |
| intersection over union | 交并比 | 可以直接说 `I-O-U` |
| generalisation | 泛化 | /ˌdʒenərəlaɪˈzeɪʃən/ |
| standard deviation | 标准差 | 常写作 standard deviation |
| convolutional | 卷积的 | /ˌkɒnvəˈluːʃənəl/ |
| annotation | 标注 | /ˌænəˈteɪʃən/ |
| threshold | 阈值 | /ˈθreʃhəʊld/ |
| overlap | 重叠 | /ˈəʊvəlæp/ |
| lightweight | 轻量的 | /ˈlaɪtweɪt/ |
| overfitting | 过拟合 | /ˌəʊvəˈfɪtɪŋ/ |
| ablation study | 消融实验 | /əˈbleɪʃən/ |

---

# Final 60-Second Summary

> In summary, I compared three approaches to 2D robotic grasp detection on the Cornell dataset. The traditional computer-vision baseline achieved 56.95 per cent. Adding Grounding DINO localisation increased the geometric pipeline to 73.33 per cent, showing that localisation provides the largest benefit.
>
> The lightweight CNN achieved an average success rate of 74.51 per cent and a higher mean IoU. On the unseen-object test set, it reached 82.35 per cent, compared with 75.3 per cent for the geometric method. However, the geometric method still produced the lowest angle error.
>
> Therefore, the VLM is effective as a localisation front end, the CNN is stronger for position and size, and the geometric method provides a useful orientation prior. The experiments are complete, and my next priority is dissertation writing and result analysis.

中文提示：

- 这是整场汇报的压缩版本。
- 如果老师临时要求“一分钟总结”，直接使用这一段。

