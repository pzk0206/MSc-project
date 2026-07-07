"""
VLM 定位实验使用的 prompt 定义。

中文说明
--------
这个文件专门放 VLM 实验用的 prompt。

为什么不把 prompt 直接写死在 run_grounding_dino_localization.py 里面？
因为 prompt 是 VLM 实验中很重要的变量：

- prompt 不同，Grounding DINO 找到的 box 可能不同；
- 论文中需要说明主实验使用的是 generic prompt 还是 user prompt；
- 把 prompt 单独放一个文件，后面改起来更安全，不会误改主算法。

本项目主要区分两类 prompt 设置：

1. generic
   所有图片都使用同一个 prompt。
   这种设置更适合和传统 CV 前端做公平对比。

2. user_prompt
   可以使用用户输入或物体类别相关的 prompt。
   这种设置能体现 VLM 的语言条件优势，但应该和 generic 设置分开报告。
"""

# Cornell 场景通常是：一个小型可抓取物体放在大桌面上。
#
# 我们前面试过：
# - "the object on the table" 容易把 table 也框进去；
# - "the foreground object" 有时也会偏大；
# - "small object" 在 Cornell 上更容易框住真正要抓的物体。
#
# 所以主实验默认使用 generic prompt = "small object"。
# 这样比较公平：所有图片都用同一句 prompt，不给每张图单独人工提示。
GENERIC_PROMPT = "small object"


def normalize_grounding_prompt(prompt: str) -> str:
    """
    统一整理 Grounding DINO 使用的 prompt 格式。

    Hugging Face 示例中通常会在 prompt 末尾加句号。
    这里自动补句号，是为了让文本格式更一致。

    例子：
        "small object" -> "small object."

    注意：
    这不是改变语义，只是把输入格式整理得更统一。
    """

    # 去掉用户不小心输入的首尾空格。
    prompt = prompt.strip()

    if not prompt:
        raise ValueError("prompt 不能为空")

    # Hugging Face / Grounding DINO 示例里常见写法是 prompt 末尾带句号。
    if not prompt.endswith("."):
        prompt = f"{prompt}."

    return prompt
