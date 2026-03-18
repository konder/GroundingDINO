#!/bin/bash
# Build GroundingDINO training data from MineStudio LMDBs (no raw video needed).
#
# Usage:
#   bash scripts/run_lmdb_pipeline.sh
#
# Assumes:
#   - NAS mounted with datasets at /mnt/nas/rocket2_train/dataset_{6xx,...}
#   - Each dataset has image/ and segmentation/ subdirs with LMDB data
set -euo pipefail

NAS_ROOT="/mnt/nas/rocket2_train"
OUTPUT_DIR="data/processed/lmdb_finetune_all"

# Collect all available dataset roots
DATA_ROOTS=()
for ds in dataset_6xx dataset_7xx dataset_8xx dataset_9xx dataset_10xx; do
    DIR="${NAS_ROOT}/${ds}"
    if [ -d "${DIR}/segmentation" ] && [ -d "${DIR}/image" ]; then
        DATA_ROOTS+=("${DIR}")
        echo "[OK] ${ds}"
    else
        echo "[SKIP] ${ds} (missing segmentation/ or image/)"
    fi
done

if [ ${#DATA_ROOTS[@]} -eq 0 ]; then
    echo "[ERROR] No valid datasets found under ${NAS_ROOT}"
    exit 1
fi

echo ""
echo "============================================"
echo " Building dataset from ${#DATA_ROOTS[@]} sources"
echo " Output: ${OUTPUT_DIR}"
echo "============================================"
echo ""

python scripts/build_lmdb_dataset.py \
    --data-root "${DATA_ROOTS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-frames 4 \
    --skip-tail 4 \
    --min-mask-area 3000 \
    --max-mask-area 0.35 \
    --visualize 50

echo ""
echo "============================================"
echo " Done! Dataset at: ${OUTPUT_DIR}"
echo "============================================"
echo ""
echo "Next: run training with:"
echo "  python scripts/train_finetune.py \\"
echo "      --config groundingdino/config/GroundingDINO_SwinT_OGC.py \\"
echo "      --pretrained weights/groundingdino_swint_ogc.pth \\"
echo "      --train-json ${OUTPUT_DIR}/annotations.json \\"
echo "      --train-images ${OUTPUT_DIR}/images/ \\"
echo "      --epochs 10 --lr 1e-5 --batch-size 4"
