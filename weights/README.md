# weights/

本目录不上传课程提供的大文件。要运行 `check.py` / `run.py`，请把课程附件中的：

```
weights/image_encoder.npz        # 152 个视觉张量，约 326 MB
weights/manifest.json
```

放到本目录下。文件 SHA-256 等信息见仓库根目录的 `SOURCE-MANIFEST.json`。

原因：该 npz 超过 GitHub 单文件 100 MB 限制，且作业明确要求不重传大参数文件。
