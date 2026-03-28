# Boundary-Aware Reproduction Guide

本文档面向当前主仓库，也就是已经复现成功的 `BEFBM + Mask2Former` 版本。现在仓库中包含两条可对比的增强路径：

- `BEFBM`：边界增强特征桥接 + 边界监督
- `Boundary Decoder HRCA`：在 `BEFBM` 基础上，进一步让边界信息参与 decoder 的 query refinement 和高分辨率 cross-attention

## 1. 环境约定

默认你已经激活了项目使用的环境：

```bash
conda activate mask2former
```

检查当前解释器和 PyTorch：

```bash
which python
python -c "import torch; print(torch.__version__)"
```

## 2. 当前实验配置

### 2.1 BEFBM 基线配置

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml`

这两份配置只打开已有的 `BOUNDARY_AWARE` 模块，不开启新的 decoder 边界细化模块。

### 2.2 BEFBM + Boundary Decoder HRCA 配置

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary_hrca.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary_hrca.yaml`

这两份配置在 `BEFBM` 基础上额外打开：

- `MODEL.MASK_FORMER.BOUNDARY_DECODER.ENABLED`
- `MODEL.MASK_FORMER.BOUNDARY_DECODER.QUERY_REFINEMENT`
- `MODEL.MASK_FORMER.BOUNDARY_DECODER.HIGH_RES_ONLY`
- `MODEL.MASK_FORMER.BOUNDARY_DECODER.TOPK_RATIO`

## 3. 新增模块说明

### 3.1 BEFBM 部分

- `mask2former/modeling/pixel_decoder/boundary_aware.py`
  - `SobelBoundaryExtractor`
  - `FeatureBridgingModule`

- `mask2former/modeling/pixel_decoder/msdeformattn.py`
  - 输出 `boundary_z_features`
  - 输出 `boundary_preds`
  - 训练态输出 `boundary_maps`
  - 保持原始 `multi_scale_features` 接口不变

### 3.2 Decoder 增强部分

- `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`
  - `BoundaryGuidedQueryRefiner`
  - `BoundaryGuidedHRCA`

设计原则：

- 不重写 Mask2Former 主框架
- 不替换数据流和训练流程
- 只让边界信息在 decoder 近端参与 query 决策

### 3.3 Head 透传

- `mask2former/modeling/meta_arch/mask_former_head.py`
  - 将 `boundary_z_features` 和 `boundary_preds` 作为 `boundary_guidance` 传入 decoder
  - 训练态保留 `boundary_maps / boundary_preds` 给 loss 使用

## 4. 推荐训练脚本

统一训练脚本：

- `scripts/train_boundary.sh`

它已经内置：

- 更干净的日志：`PYTHONWARNINGS=ignore::FutureWarning`
- 更稳的显存分配：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- 自动根据 `--gpus` 推导 `--num-gpus`
- 自动生成时间戳日志到 `logs/`
- 支持前台训练、后台训练、评估

先确保脚本可执行：

```bash
chmod +x scripts/train_boundary.sh
```

查看帮助：

```bash
scripts/train_boundary.sh --help
```

## 5. 训练流程

### 5.1 训练 BEFBM 基线

R50:

```bash
scripts/train_boundary.sh --model r50 --variant befbm
```

R101:

```bash
scripts/train_boundary.sh --model r101 --variant befbm
```

### 5.2 训练 BEFBM + Boundary Decoder HRCA

R50:

```bash
scripts/train_boundary.sh --model r50 --variant hrca
```

R101:

```bash
scripts/train_boundary.sh --model r101 --variant hrca
```

### 5.3 单卡训练

如果要更稳妥地控制显存：

```bash
scripts/train_boundary.sh --model r50 --variant hrca --gpus 2 --ims-per-batch 2 --base-lr 0.000025
```

### 5.4 双卡训练

```bash
scripts/train_boundary.sh --model r50 --variant hrca --gpus 2,3
```

## 6. 后台训练与日志

如果你希望本地电脑关闭后服务器继续训练，直接用：

```bash
scripts/train_boundary.sh --model r50 --variant hrca --background
```

脚本会输出：

- 后台进程 PID
- 日志文件路径
- 查看日志命令

例如查看实时日志：

```bash
tail -f logs/r50_hrca_gpu2_3_train_*.log
```

查看训练进程：

```bash
ps -ef | rg train_net.py
```

查看 GPU：

```bash
nvidia-smi
```

## 7. 断线续训

如果上一次训练目录里已经有 checkpoint：

```bash
scripts/train_boundary.sh --model r50 --variant hrca --resume
```

## 8. 评估流程

### 8.1 评估 HRCA 模型

```bash
scripts/train_boundary.sh \
  --model r50 \
  --variant hrca \
  --mode eval \
  --weights output/model_final.pth
```

### 8.2 评估 BEFBM 基线模型

```bash
scripts/train_boundary.sh \
  --model r50 \
  --variant befbm \
  --mode eval \
  --weights output/model_final.pth
```

如果你要指定单卡评估：

```bash
scripts/train_boundary.sh \
  --model r50 \
  --variant hrca \
  --mode eval \
  --weights output/model_final.pth \
  --gpus 2
```

## 9. 常用对比实验

### 9.1 BEFBM vs HRCA

保持 backbone 一致，只切换 `--variant`：

```bash
scripts/train_boundary.sh --model r50 --variant befbm
scripts/train_boundary.sh --model r50 --variant hrca
```

### 9.2 R50 vs R101

保持模块一致，只切换 `--model`：

```bash
scripts/train_boundary.sh --model r50 --variant hrca
scripts/train_boundary.sh --model r101 --variant hrca
```

### 9.3 调整 HRCA 稀疏度

通过附加 detectron2 配置项覆盖：

```bash
scripts/train_boundary.sh --model r50 --variant hrca -- \
  MODEL.MASK_FORMER.BOUNDARY_DECODER.TOPK_RATIO 64
```

`TOPK_RATIO` 越小，保留的高边界 token 越多，显存和计算也会更高。

## 10. 输出兼容性

当前实现保持以下兼容性：

- 不修改 dataset / dataloader / train entry / eval pipeline
- 推理输出字段保持 baseline 兼容
- 语义分割任务下推理结果仍为 `sem_seg`
- `BOUNDARY_DECODER.ENABLED=False` 时可退回原始 `BEFBM + Mask2Former`

## 11. 代码入口

如果你要继续调模块，优先看这些文件：

- `mask2former/config.py`
- `mask2former/modeling/pixel_decoder/msdeformattn.py`
- `mask2former/modeling/meta_arch/mask_former_head.py`
- `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`
- `scripts/train_boundary.sh`
