# 数据目录说明

```
data/
├── test/              # 手工标注的 Minecraft 测试集（已入库）
│   ├── *.png          # 测试图片（16 张，覆盖挖矿/狩猎/建造等场景）
│   └── annotations.json
├── raw/               # 原始数据（不入库）
│   └── minestudio/    # MineStudio LMDB 数据
├── processed/         # 预处理后的训练数据（不入库）
│   └── coco/          # 转换后的 COCO 格式数据
├── lmdb/              # LMDB 缓存（不入库）
├── cache/             # HuggingFace 等下载缓存（不入库）
└── README.md
```

## 数据流向

1. `raw/` ← 从 HuggingFace 下载的 MineStudio LMDB 原始数据
2. `processed/` ← 通过 `scripts/` 中的转换脚本生成 COCO 格式训练数据
3. `test/` ← 手工标注的评估数据，随代码版本管理

## 注意事项

- `raw/`、`processed/`、`lmdb/`、`cache/` 目录下的数据文件已在 `.gitignore` 中排除
- 仅 `test/` 和本 `README.md` 受版本管理
- 大文件请勿直接提交到 git
