#!/bin/bash
# Full data generation + merge pipeline for production machine.
#
# Usage:
#   bash scripts/run_full_pipeline.sh
#
# Assumes:
#   - NAS mounted at /mnt/nas/
#   - Datasets at /mnt/nas/rocket2_train/dataset_{6xx,7xx,8xx,9xx,10xx}/
#   - Raw videos at /mnt/nas/vpt_raw_video/
#   - Python environment with all dependencies
set -euo pipefail

NAS_ROOT="/mnt/nas/rocket2_train"
RAW_VIDEO="/mnt/nas/vpt_raw_video"
OUTPUT_BASE="data/processed"
MERGED_DIR="${OUTPUT_BASE}/finetune_merged"

DATASETS=("dataset_6xx" "dataset_7xx" "dataset_8xx" "dataset_9xx" "dataset_10xx")

MIN_MASK_AREA=5000
MAX_MASK_AREA=0.35
VIZ_SAMPLES=30

echo "============================================"
echo " GroundingDINO Minecraft Fine-tune Pipeline"
echo "============================================"
echo ""

# --- Step 1: Generate per-dataset ---
GENERATED_DIRS=()
for ds in "${DATASETS[@]}"; do
    DATA_ROOT="${NAS_ROOT}/${ds}"
    OUT_DIR="${OUTPUT_BASE}/finetune_${ds}"

    if [ ! -d "${DATA_ROOT}/segmentation" ]; then
        echo "[SKIP] ${DATA_ROOT}/segmentation not found"
        continue
    fi

    echo ""
    echo "========== Processing ${ds} =========="
    python scripts/build_finetune_dataset.py \
        --data-root "${DATA_ROOT}" \
        --output-dir "${OUT_DIR}" \
        --raw-video-dir "${RAW_VIDEO}" \
        --min-mask-area ${MIN_MASK_AREA} \
        --max-mask-area ${MAX_MASK_AREA} \
        --visualize ${VIZ_SAMPLES}

    GENERATED_DIRS+=("${OUT_DIR}")
    echo "[DONE] ${ds} → ${OUT_DIR}"
done

# --- Step 2: Merge all ---
if [ ${#GENERATED_DIRS[@]} -eq 0 ]; then
    echo "[ERROR] No datasets generated!"
    exit 1
fi

echo ""
echo "========== Merging ${#GENERATED_DIRS[@]} datasets =========="
python scripts/merge_coco_datasets.py \
    --inputs "${GENERATED_DIRS[@]}" \
    --output "${MERGED_DIR}" \
    --symlinks

echo ""
echo "========== Quality evaluation =========="
python scripts/evaluate_data_quality.py \
    --dataset-dir "${MERGED_DIR}" \
    --samples-per-cat 9

echo ""
echo "============================================"
echo " Pipeline complete!"
echo " Merged dataset: ${MERGED_DIR}"
echo " Categories + images: see ${MERGED_DIR}/annotations.json"
echo " Quality report: ${MERGED_DIR}/quality_report/"
echo "============================================"
echo ""
echo "Next: run training with:"
echo "  python scripts/train_finetune.py \\"
echo "      --config groundingdino/config/GroundingDINO_SwinT_OGC.py \\"
echo "      --pretrained weights/groundingdino_swint_ogc.pth \\"
echo "      --train-json ${MERGED_DIR}/annotations.json \\"
echo "      --train-images ${MERGED_DIR}/images/ \\"
echo "      --epochs 10 --lr 1e-5 --batch-size 4"
