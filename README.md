# Project 1｜用 NumPy 复现 OpenCLIP ViT-B-32 图像编码器

课程：Python 程序设计　|　深圳大学　|　Project 1

**仅用 NumPy 实现课程微调版 OpenCLIP ViT-B-32 的图像塔前向计算**，不依赖 PyTorch / OpenCLIP / GPU / 网络，不含任何训练过程。给定 152 个已训练好的 float32 权重张量，把 `[B,3,224,224]` 的预处理图像映射为 `[B,512]` 的单位长度输出向量，并与课程提供的独立 FP32 参考结果对齐。

## 成果

`python check.py --stage all` **8 个阶段全部 PASS**：

| 阶段 | 形状 | max_abs | mean_abs | 容差 atol |
|---|---|---|---|---|
| preprocess | (10, 3, 224, 224) | 0.000e+00 | 0.000e+00 | 2e-6 |
| patches | (2, 49, 3072) | 0.000e+00 | 0.000e+00 | 2e-4 |
| tokens | (2, 50, 768) | 4.005e-05 | 1.289e-06 | 2e-4 |
| block0 | (2, 50, 768) | 7.063e-06 | 3.250e-07 | 2e-4 |
| last_tokens | (2, 50, 768) | 4.768e-05 | 2.696e-06 | 2e-4 |
| output | (10, 512) | 5.439e-07 | 8.311e-08 | 2e-5 |
| batch | (3, 512) | 4.619e-07 | 7.632e-08 | 2e-5 |

输出为单位长度 float32 向量，长度与 1 的最大偏差约 1.2e-07。另用未参与参考集的图片验证，`run.py --images` 输出 `(1,512)` float32、数值有限、单位长度偏差 0。

## 目录结构

```
.
├── image_encoder.py          ★ 本人实现：六处 TODO 全部补全，唯一需要修改的文件
├── helpers.py                （课程提供）LayerNorm / 精确 GELU / softmax / 单位化
├── preprocess_image.py       （课程提供）缩放、中心裁剪与归一化
├── runtime.py / run.py       （课程提供）加载器与命令行
├── check.py                  （课程提供）8 阶段数值验收
├── transformer_steps.py      （课程提供）迷你版教学示例
├── config.json               （课程提供）模型结构常数
├── requirements.txt
├── PPT-CODE-MAP.txt          （课程提供）课件与代码行号对照
├── SOURCE-MANIFEST.json      （课程提供）文件来源与校验信息
├── weights/README.md         说明：需放入课程附件的 image_encoder.npz
├── assets/README.md          说明：需放入课程附件的图片与参考数据
└── report/
    ├── 实验报告-Project1-NumPy复现ViT-B-32.docx   实验报告（深圳大学模板）
    ├── 报告正文.md                                报告 Markdown 源稿
    ├── README.md                                  提交说明与运行命令
    ├── 架构图-ProcessOn输入.md                    绘图需求与 Mermaid 源码
    └── figures/
        ├── fig1_主干.png                          主干流水线（A4 排版版）
        └── fig2_block.png                         单个 Transformer Block（A4 排版版）
```

## 快速开始

```bash
# 1. 安装依赖（Python 3.9+）
python -m pip install -r requirements.txt

# 2. 从课程附件补齐本仓库未包含的大文件（见下方说明）
#    weights/image_encoder.npz
#    assets/images/*.jpg、assets/images.json、assets/manifest.json
#    assets/reference_outputs.npz

# 3. 数值验收与推理
python check.py --stage all
python run.py --output output_tokens.npz
python run.py --images your.jpg --output my_tokens.npz
```

分阶段调试（按依赖顺序，失败时先修最早出错的阶段）：

```bash
python check.py --stage patches && python check.py --stage tokens \
  && python check.py --stage block0 && python check.py --stage last_tokens \
  && python check.py --stage output && python check.py --stage batch
```

## 为什么仓库里没有权重和参考数据

课程提供的 `weights/image_encoder.npz` 约 326 MB、`assets/reference_outputs.npz` 约 7.6 MB，既超出 GitHub 单文件 100 MB 限制，也不属于作业要求提交的内容（作业明确要求「不重传大参数文件与参考数据」）。因此本仓库只包含代码、报告与图；验收所需的权重与数据请从课程附件获取，放入 `weights/` 与 `assets/` 后即可完整运行 `check.py`。`transformer_steps.py` 与课堂演示不依赖这些文件，可独立运行。

## 实现要点

| TODO | 方法 | 关键约定 |
|---|---|---|
| 01 | `patchify` | `[B,3,224,224]` → reshape `[B,3,7,32,7,32]` → transpose(0,2,4,1,3,5) → `[B,49,3072]`；patch 行优先、块内 C,H,W |
| 02 | `split_heads` | 先 `[B,N,12,64]` 再 transpose 成 `[B,12,N,64]`，直接 reshape 会交错 token |
| 03 | `MultiHeadAttention.__call__` | `q @ k.swapaxes(-1,-2) / sqrt(64)`，softmax → 加权求和 → 合并头 → 输出投影 |
| 04 | `TransformerBlock.__call__` | 两次残差 + GELU MLP，无 LayerScale |
| 05 | `embed` | `patches @ patch_weight.T`（无 bias）+ CLS + 位置嵌入 |
| 06 | `__call__` | 取 CLS → post_norm → `visual.proj` → 单位化 |

三个必须记住的约定：

1. 线性层权重按 `[out,in]` 存放，统一 `x @ W.T + b`；**唯一例外** `visual.proj` 已是 `[in,out]=[768,512]`，不转置。
2. 全程 float32。缩放因子写 `np.sqrt(np.float32(64))`：`np.sqrt(64)` 是 float64 标量，在 NumPy 2 下会把 scores 提升为 float64，触发 `check.py` 的 dtype 断言。
3. 拆头与 4 维 k 的转置不能照搬二维教学示例（用 `swapaxes(-1,-2)` 而非 `k.T`）。

## 报告

完整实验报告见 `report/实验报告-Project1-NumPy复现ViT-B-32.docx`，采用学校实验报告模板，含封面、四章节、误差表与两张架构图；正文 4 页（另有封面）。

## 声明

六个 TODO 的设计与实现、形状推导、数值验收、架构图与报告撰写由本人完成。AI 助手用于作业材料梳理、实现方案讨论、图示排版与文稿修改；最终代码与结论均经本人运行验证。课程提供的文件（见目录标注）未作修改。
