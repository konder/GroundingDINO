# GroundingDINO for Minecraft — 项目目标

## 项目概述

基于 [GroundingDINO](https://arxiv.org/abs/2303.05499) 开放集目标检测模型，通过微调使其在 Minecraft 游戏画面上具备高精度的目标检测能力。

## 核心目标

利用 [MineStudio VPT 训练数据](https://huggingface.co/datasets/CraftJarvis/minestudio-data-6xx-v106) 等 Minecraft 领域数据集，对 GroundingDINO 进行领域微调（domain fine-tuning），提升模型对 Minecraft 画面中各类实体（生物、方块、物品、建筑等）的检测准确率。

## 数据来源

| 数据集 | 描述 | 格式 |
|--------|------|------|
| [CraftJarvis/minestudio-data-6xx-v106](https://huggingface.co/datasets/CraftJarvis/minestudio-data-6xx-v106) | MineStudio VPT 训练数据，包含 event、action、video、segment 等子集 | LMDB |
| `data/test/` | 手工标注的 Minecraft 测试集（16 张图片，覆盖挖矿、狩猎、建造等场景） | PNG + JSON |

## 微调策略

1. **基础模型**: GroundingDINO-T (Swin-T backbone) 预训练权重
2. **训练数据**: 从 MineStudio 数据集中提取视频帧 + 事件标注，构建 COCO 格式的检测数据集
3. **目标类别**: Minecraft 特有实体（矿石、生物、方块、工具、建筑结构等）
4. **评估指标**: 在 `data/test/` 测试集上的 mAP、各类别 AP

## 里程碑

- [ ] 数据集评估：分析 MineStudio event/data.mdb 的数据结构和可用性
- [ ] 数据管线：构建从 LMDB 到 COCO 格式的数据转换管线
- [ ] 训练管线：实现 GroundingDINO 微调训练代码
- [ ] 基线评估：在 Minecraft 测试集上评估原始模型表现
- [ ] 微调训练：完成首轮微调并对比评估
- [ ] 推理部署：优化推理速度，支持实时 Minecraft 画面检测

## 技术栈

- **模型**: GroundingDINO (Swin-T + BERT + Deformable DETR)
- **框架**: PyTorch
- **环境**: conda `dino` 环境
- **数据格式**: LMDB (MineStudio) → COCO JSON (训练用)
