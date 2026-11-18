# Project 1 提交说明 — NumPy 复现 OpenCLIP ViT-B-32 图像编码器

本人完成的文件：`image_encoder.py`（补全 TODO-01 ~ TODO-06）。
其余文件（`preprocess_image.py`、`helpers.py`、`runtime.py`、`run.py`、`check.py`、`transformer_steps.py`、权重与参考数据）均为课程提供，未做任何修改。

代码仓库：https://github.com/66LzC66/project1-numpy-vit-encoder

## 环境

- Python 3.9 及以上（本机使用 3.13 验证）
- 仅依赖 `requirements.txt` 中的三个包：numpy、Pillow、imageio
- 不需要 GPU、PyTorch、OpenCLIP 包或网络

```bash
python -m pip install -r requirements.txt
```

## 运行方式

在本目录（`project1-numpy-student/`）下执行：

```bash
# 全部数值验收（资源 → 预处理 → patch → 嵌入 → 第1层 → 12层 → 输出 → 乱序批）
python check.py --stage all

# 用提供的 10 张图片生成输出向量
python run.py --output output_tokens.npz

# 用任意自有图片（验证泛化，支持 jpg/png，可多张）
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz

# 分阶段验收（调试用，按依赖顺序）
python check.py --stage patches && python check.py --stage tokens \
  && python check.py --stage block0 && python check.py --stage last_tokens \
  && python check.py --stage output && python check.py --stage batch
```

`run.py` 输出 `npz`，含 `ids` 与 `output_tokens`（float32，形状 `[B,512]`，每行为单位长度向量）。

## 验收结果（本机实测）

| 阶段 | 形状 | max_abs | mean_abs | 容差 (atol, rtol) |
|---|---|---|---|---|
| preprocess | (10, 3, 224, 224) | 0.000e+00 | 0.000e+00 | 2e-6, 2e-4 |
| patches | (2, 49, 3072) | 0.000e+00 | 0.000e+00 | 2e-4, 2e-4 |
| tokens | (2, 50, 768) | 4.005e-05 | 1.289e-06 | 2e-4, 2e-4 |
| block0 | (2, 50, 768) | 7.063e-06 | 3.250e-07 | 2e-4, 2e-4 |
| last_tokens | (2, 50, 768) | 4.768e-05 | 2.696e-06 | 2e-4, 2e-4 |
| output | (10, 512) | 5.439e-07 | 8.311e-08 | 2e-5, 2e-4 |
| batch | (3, 512) | 4.619e-07 | 7.632e-08 | 2e-5, 2e-4 |

8 个阶段全部 PASS；输出向量单位长度最大偏差约 1.2e-07（float32 精度极限）。

## 实现要点

- `patchify`：`[B,3,224,224] → reshape [B,3,7,32,7,32] → transpose(0,2,4,1,3,5) → reshape [B,49,3072]`，patch 行优先，块内按 C,H,W 拍平
- `split_heads`：先 `[B,N,768] → [B,N,12,64]`，再 `transpose(0,2,1,3) → [B,12,N,64]`
- 注意力：`scores = q @ k.swapaxes(-1,-2) / sqrt(64)`（除以 `np.float32` 标量，避免 NumPy 2 提升为 float64）；`context = softmax(scores) @ v`；合并头后 `merged @ out_weight.T + out_bias`
- block：`residual = x + attn(norm1(x))`；`u = norm2(residual)`；`residual + gelu(u @ W1.T + b1) @ W2.T + b2`
- `embed`：`patches @ patch_weight.T`（无 bias）→ CLS 拼接 → 加位置嵌入
- 输出头：`tokens[:,0,:] → post_norm → @ visual.proj`（该权重已是 `[in,out]`，不转置）→ 单位化

详细分析见报告。
