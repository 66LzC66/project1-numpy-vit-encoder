# assets/

本目录不上传课程提供的图片与参考结果。要完整运行 `check.py`，请把课程附件中的：

```
assets/images/image_01.jpg ... image_10.jpg
assets/images.json
assets/manifest.json
assets/reference_outputs.npz    # 独立 FP32 参考，约 7.6 MB
```

放到本目录下。参考数组只作为 `check.py` 的对比目标，绝不是编码器的输入。

原因：作业明确要求不重传参考数据；同时避免仓库体积膨胀。
