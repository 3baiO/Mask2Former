# Boundary-Aware Semantic Segmentation Reproduction (Mask2Former Baseline)

本文档说明本仓库中 Boundary-Aware 复现的主要改动，以及如何在 `conda` 环境 `mask2former` 下进行训练和测试。

## 1. 解释器与环境

项目测试默认使用：

```bash
conda activate mask2former
```

例如检查解释器中的 PyTorch：

```bash
python -c "import torch; print(torch.__version__)"
```

## 2. 主要改动概览

### 2.1 新增模块

- `mask2former/modeling/pixel_decoder/boundary_aware.py`
  - `SobelBoundaryExtractor`
  - `FeatureBridgingModule`

### 2.2 集成位置与数据流

- `mask2former/modeling/pixel_decoder/msdeformattn.py`
  - 在 `input_proj` 后、deformable encoder 前接入 BEFBM
  - 训练态额外产生：
    - 多尺度 `boundary_z_features`（`Z^l`）
    - 多尺度 `boundary_maps`（`E^l`）
    - 多尺度 `boundary_preds`（由 `1x1 conv` 从 `Z^l` 映射到 1 通道）
    - `boundary_alphas`

- `mask2former/modeling/meta_arch/mask_former_head.py`
  - 透传 `images` 到 pixel decoder（若 decoder 支持）
  - 训练态将 boundary 分支结果加到输出 dict

- `mask2former/maskformer_model.py`
  - 训练态传入反归一化后的 `boundary_images`
  - criterion 开启时合并 `loss_edge`
  - debug 模式打印 `Z^l/E^l` shape、`alpha` 统计、各项 loss

### 2.3 损失与配置

- `mask2former/modeling/criterion.py`
  - 新增 `loss_edge`
  - 公式实现：按尺度求 MSE 后求和

- `mask2former/config.py`
  - 新增配置节点 `MODEL.MASK_FORMER.BOUNDARY_AWARE`
  - 默认：
    - `ENABLED: False`
    - `TO_GRAYSCALE: True`
    - `NORMALIZE: True`
    - `EDGE_LOSS_WEIGHT: 0.1`
    - `DEBUG: False`

- `train_net.py`
  - 启动时记录 boundary-aware 配置
  - 对明显误配给出 warning（如开启模块但 `EDGE_LOSS_WEIGHT <= 0`）

### 2.4 新增训练配置文件

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml`

这两个配置在原 baseline 基础上仅打开 `BOUNDARY_AWARE` 开关，不改变推理输出格式。

## 3. 训练步骤

### 3.0 推荐：使用统一训练脚本（免手动改长命令）

新增脚本：`scripts/train_boundary.sh`

常用用法：

```bash
# 默认：R50 boundary 配置，GPU 2,3，IMS_PER_BATCH=4，BASE_LR=5e-5
scripts/train_boundary.sh

# 单卡更稳妥（GPU 2）
scripts/train_boundary.sh --gpus 2 --ims-per-batch 2 --base-lr 0.000025

# 后台运行并写日志（断开 SSH 后继续）
scripts/train_boundary.sh --background --name r50_ba_gpu23
```

脚本内已内置：
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（减少显存碎片问题）
- `PYTHONWARNINGS=ignore::FutureWarning`（让训练日志更干净）
- 自动按 `--gpus` 计算 `--num-gpus`
- 自动创建并写入 `logs/*.log`

### 3.1 Boundary-Aware 训练（Cityscapes, R50）

```bash
python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml \
  --num-gpus 1
```

### 3.2 Boundary-Aware 训练（Cityscapes, R101）

```bash
python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml \
  --num-gpus 1
```

### 3.3 动态覆盖关键配置（示例）

```bash
python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml \
  --num-gpus 1 \
  MODEL.MASK_FORMER.BOUNDARY_AWARE.EDGE_LOSS_WEIGHT 0.2 \
  MODEL.MASK_FORMER.BOUNDARY_AWARE.DEBUG True
```

## 4. 测试/评估步骤

使用 `--eval-only`，并指定权重路径：

```bash
C
```

R101 同理替换配置文件。

## 5. Baseline 对照（关闭 Boundary-Aware）

若需要与原始 baseline 对照，直接使用原配置：

```bash
python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml \
  --num-gpus 1
```

或在 boundary 配置上强制关闭：

```bash
python train_net.py \
  --config-file configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml \
  --num-gpus 1 \
  MODEL.MASK_FORMER.BOUNDARY_AWARE.ENABLED False
```

## 6. 结果兼容性说明

- 推理阶段不计算 `E`
- 推理输出字段与 baseline 保持一致（如语义分割配置下输出 `sem_seg`）
- 开关关闭时不引入 `loss_edge`，行为可退化为 baseline 路径
