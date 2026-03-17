"""
Fine-tune GroundingDINO on Minecraft COCO-format dataset.

Core design:
  - Loads a pretrained GroundingDINO model and freezes backbone + encoder
  - Only trains: decoder layers, bbox_embed, class_embed, feat_map, input_proj
  - Uses Hungarian matching (scipy) to assign GT boxes to predictions
  - Loss = sigmoid focal (classification) + L1 (bbox) + GIoU (bbox)
  - DataLoader reads COCO annotations.json + images/
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.misc import clean_state_dict, nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.vl_utils import create_positive_map_from_span


# ---------------------------------------------------------------------------
# Caption / token-span utilities
# ---------------------------------------------------------------------------

def build_caption_and_spans(
    categories: List[dict],
) -> Tuple[str, Dict[int, List[List[int]]]]:
    """Build a single caption string and per-category char-level spans.

    Returns:
        caption:  e.g. "stone . coal ore . granite ."
        id2span:  {cat_id: [[start, end], ...]}
    """
    caption = ""
    id2span: Dict[int, List[List[int]]] = {}
    for cat in categories:
        name = cat["name"].replace("_", " ").lower()
        tokens_positive = []
        for word in name.split():
            if caption:
                caption += " "
            start = len(caption)
            end = start + len(word)
            tokens_positive.append([start, end])
            caption += word
        caption += " ."
        id2span[cat["id"]] = tokens_positive
    return caption, id2span


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _make_train_transform():
    return T.Compose([
        T.RandomResize([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                        max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class MinecraftCocoDataset(Dataset):
    """COCO-format dataset for GroundingDINO fine-tuning."""

    def __init__(self, ann_path: str, img_dir: str, max_text_len: int = 256):
        from transformers import AutoTokenizer

        with open(ann_path) as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.max_text_len = max_text_len
        self.transform = _make_train_transform()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.images = {img["id"]: img for img in data["images"]}
        self.categories = data["categories"]
        self.caption, self.id2span = build_caption_and_spans(self.categories)
        self._tokenized = self.tokenizer(self.caption, return_tensors="pt")

        img2anns: Dict[int, list] = {}
        for ann in data["annotations"]:
            img2anns.setdefault(ann["image_id"], []).append(ann)

        self.samples = []
        n_missing = 0
        for img_id in sorted(self.images.keys()):
            if not img2anns.get(img_id):
                continue
            fp = os.path.join(img_dir, self.images[img_id]["file_name"])
            if not os.path.isfile(fp):
                n_missing += 1
                continue
            self.samples.append((img_id, img2anns[img_id]))

        if n_missing:
            _print(f"[dataset] Skipped {n_missing} images (files not found)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_id, anns = self.samples[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        boxes_xyxy = []
        cat_ids = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes_xyxy.append([x, y, x + bw, y + bh])
            cat_ids.append(ann["category_id"])

        boxes_xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)

        target = {"boxes": boxes_xyxy}
        image, target = self.transform(image, target)
        boxes_cxcywh = target["boxes"]  # cxcywh, normalized by Normalize transform

        spans_per_box = []
        for cid in cat_ids:
            spans_per_box.append(self.id2span[cid])
        positive_map = create_positive_map_from_span(
            self._tokenized, spans_per_box, max_text_len=self.max_text_len
        )

        return {
            "image": image,
            "boxes": boxes_cxcywh,
            "labels": torch.tensor(cat_ids, dtype=torch.long),
            "positive_map": positive_map,
            "caption": self.caption,
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> Tuple[torch.Tensor, List[dict]]:
    images = [item["image"] for item in batch]
    targets = [
        {
            "boxes": item["boxes"],
            "positive_map": item["positive_map"],
            "labels": item["labels"],
        }
        for item in batch
    ]
    captions = [item["caption"] for item in batch]
    return images, targets, captions


# ---------------------------------------------------------------------------
# Hungarian Matcher
# ---------------------------------------------------------------------------

@torch.no_grad()
def hungarian_match(
    outputs: dict,
    targets: List[dict],
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Match predictions to GT using the Hungarian algorithm.

    Returns list of (pred_indices, gt_indices) per batch element.
    """
    bs = outputs["pred_logits"].shape[0]
    indices = []

    for b in range(bs):
        pred_logits = outputs["pred_logits"][b].sigmoid()  # (N, C)
        pred_boxes = outputs["pred_boxes"][b]  # (N, 4)
        gt_boxes = targets[b]["boxes"].to(pred_boxes.device)  # (M, 4)
        gt_positive_map = targets[b]["positive_map"].to(pred_logits.device)  # (M, C)

        if gt_boxes.shape[0] == 0:
            indices.append(
                (torch.tensor([], dtype=torch.long),
                 torch.tensor([], dtype=torch.long))
            )
            continue

        # Classification cost: per-query score for each GT category
        # gt_positive_map: (M, C), pred_logits: (N, C) → (N, M)
        cost_cls = -(pred_logits @ gt_positive_map.t())

        # L1 bbox cost
        cost_l1 = torch.cdist(pred_boxes, gt_boxes, p=1)

        # GIoU cost
        cost_g = -box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_boxes),
            box_ops.box_cxcywh_to_xyxy(gt_boxes),
        )

        C = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_g
        C = C.cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(C)
        indices.append(
            (torch.tensor(row_ind, dtype=torch.long),
             torch.tensor(col_ind, dtype=torch.long))
        )

    return indices


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    outputs: dict,
    targets: List[dict],
    cost_class: float = 2.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict:
    """Compute the training loss after Hungarian matching."""
    indices = hungarian_match(
        outputs, targets,
        cost_class=cost_class,
        cost_bbox=cost_bbox,
        cost_giou=cost_giou,
    )

    device = outputs["pred_logits"].device
    num_boxes = max(sum(len(t["boxes"]) for t in targets), 1)

    # --- Classification loss (sigmoid focal) ---
    pred_logits_list = []
    target_classes_list = []
    max_text_len = outputs["pred_logits"].shape[-1]

    for b, (pred_idx, gt_idx) in enumerate(indices):
        logits = outputs["pred_logits"][b]  # (N, C)
        tgt = torch.zeros_like(logits)
        if len(pred_idx) > 0:
            gt_pm = targets[b]["positive_map"].to(device)  # (M, C)
            # Pad or truncate to match pred logits dim
            if gt_pm.shape[1] < max_text_len:
                gt_pm = F.pad(gt_pm, (0, max_text_len - gt_pm.shape[1]))
            elif gt_pm.shape[1] > max_text_len:
                gt_pm = gt_pm[:, :max_text_len]
            tgt[pred_idx] = gt_pm[gt_idx]
        pred_logits_list.append(logits)
        target_classes_list.append(tgt)

    pred_logits_cat = torch.stack(pred_logits_list)  # (B, N, C)
    target_cat = torch.stack(target_classes_list)  # (B, N, C)

    loss_ce = _sigmoid_focal_loss(
        pred_logits_cat.flatten(0, 1),
        target_cat.flatten(0, 1),
        num_boxes,
        alpha=focal_alpha,
        gamma=focal_gamma,
    )

    # --- Bbox losses (L1 + GIoU) ---
    loss_bbox = torch.tensor(0.0, device=device)
    loss_giou = torch.tensor(0.0, device=device)

    src_boxes_list = []
    tgt_boxes_list = []
    for b, (pred_idx, gt_idx) in enumerate(indices):
        if len(pred_idx) == 0:
            continue
        src = outputs["pred_boxes"][b][pred_idx]
        tgt = targets[b]["boxes"].to(device)[gt_idx]
        src_boxes_list.append(src)
        tgt_boxes_list.append(tgt)

    if src_boxes_list:
        src_boxes = torch.cat(src_boxes_list)
        tgt_boxes = torch.cat(tgt_boxes_list)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="sum") / num_boxes

        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(tgt_boxes)
        src_xyxy = src_xyxy.clamp(min=0, max=1)
        tgt_xyxy = tgt_xyxy.clamp(min=0, max=1)
        giou = box_ops.generalized_box_iou(src_xyxy, tgt_xyxy)
        loss_giou = (1 - torch.diag(giou)).sum() / num_boxes

    return {
        "loss_ce": loss_ce,
        "loss_bbox": loss_bbox * cost_bbox,
        "loss_giou": loss_giou * cost_giou,
    }


def _sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    inputs = inputs.clamp(-50, 50)
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Freeze strategy
# ---------------------------------------------------------------------------

def apply_freeze_strategy(model: torch.nn.Module, strategy: str = "decoder_only"):
    """Freeze parameters based on strategy.

    Strategies:
        - decoder_only: freeze backbone + encoder + bert; train decoder + heads
        - heads_only:   freeze everything except bbox_embed, class_embed, feat_map
        - full:         train everything (no freezing)
    """
    if strategy == "full":
        return

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_prefixes = []
    if strategy == "decoder_only":
        trainable_prefixes = [
            "transformer.decoder",
            "bbox_embed",
            "class_embed",
            "feat_map",
            "input_proj",
            "transformer.tgt_embed",
            "transformer.refpoint_embed",
            "transformer.enc_out_bbox_embed",
            "transformer.enc_out_class_embed",
        ]
    elif strategy == "heads_only":
        trainable_prefixes = [
            "bbox_embed",
            "class_embed",
            "feat_map",
        ]
    else:
        raise ValueError(f"Unknown freeze strategy: {strategy}")

    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in trainable_prefixes):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _print(f"[freeze] strategy={strategy}: trainable {trainable:,} / {total:,} "
           f"({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    max_grad_norm: float = 0.1,
    log_interval: int = 20,
) -> dict:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_bbox = 0.0
    total_giou = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (images, targets, captions) in enumerate(dataloader):
        samples = nested_tensor_from_tensor_list(
            [img.to(device) for img in images]
        )
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["positive_map"] = t["positive_map"].to(device)

        outputs = model(samples, captions=captions)

        loss_dict = compute_loss(outputs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
        optimizer.step()

        total_loss += loss.item()
        total_ce += loss_dict["loss_ce"].item()
        total_bbox += loss_dict["loss_bbox"].item()
        total_giou += loss_dict["loss_giou"].item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            avg = total_loss / n_batches
            _print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(dataloader)} "
                f"| loss={avg:.4f} (ce={total_ce / n_batches:.4f} "
                f"bbox={total_bbox / n_batches:.4f} "
                f"giou={total_giou / n_batches:.4f}) "
                f"| {elapsed:.1f}s"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_ce": total_ce / max(n_batches, 1),
        "loss_bbox": total_bbox / max(n_batches, 1),
        "loss_giou": total_giou / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for images, targets, captions in dataloader:
        samples = nested_tensor_from_tensor_list(
            [img.to(device) for img in images]
        )
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["positive_map"] = t["positive_map"].to(device)

        outputs = model(samples, captions=captions)
        loss_dict = compute_loss(outputs, targets)
        total_loss += sum(loss_dict.values()).item()
        n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GroundingDINO on Minecraft")
    parser.add_argument("--config", required=True, help="Model config .py path")
    parser.add_argument("--pretrained", required=True, help="Pretrained checkpoint .pth")
    parser.add_argument("--train-json", required=True, help="COCO annotations.json")
    parser.add_argument("--train-images", required=True, help="Image directory")
    parser.add_argument("--val-json", default=None, help="Validation annotations.json")
    parser.add_argument("--val-images", default=None, help="Validation image directory")
    parser.add_argument("--output-dir", default="outputs/finetune", help="Checkpoint output dir")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=8, help="Epoch to drop LR")
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--freeze", default="decoder_only",
                        choices=["decoder_only", "heads_only", "full"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction for validation split if --val-json not given")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            _print("[device] CUDA not available, using MPS")
        else:
            device = "cpu"
            _print("[device] CUDA not available, using CPU")

    # --- Load model ---
    _print(f"[model] Loading config from {args.config}")
    model_args = SLConfig.fromfile(args.config)
    model_args.device = device
    model = build_model(model_args)

    _print(f"[model] Loading pretrained weights from {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location="cpu")
    model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.to(device)

    apply_freeze_strategy(model, args.freeze)

    # --- Dataset ---
    full_dataset = MinecraftCocoDataset(args.train_json, args.train_images)
    _print(f"[data] Loaded {len(full_dataset)} training samples, "
           f"caption: \"{full_dataset.caption[:80]}...\"")

    if args.val_json and args.val_images:
        train_dataset = full_dataset
        val_dataset = MinecraftCocoDataset(args.val_json, args.val_images)
    else:
        n_val = max(1, int(len(full_dataset) * args.val_split))
        n_train = len(full_dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        _print(f"[data] Auto-split: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    # --- Optimizer + Scheduler ---
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_drop, gamma=0.1
    )

    # --- Training ---
    best_val_loss = float("inf")
    history = []
    _print(f"\n{'='*60}")
    _print(f"Starting fine-tuning: {args.epochs} epochs, "
           f"batch_size={args.batch_size}, lr={args.lr}")
    _print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch,
            max_grad_norm=args.max_grad_norm,
            log_interval=args.log_interval,
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        epoch_time = time.time() - t_start
        lr_now = optimizer.param_groups[0]["lr"]
        _print(
            f"Epoch {epoch}/{args.epochs} "
            f"| train_loss={train_metrics['loss']:.4f} "
            f"| val_loss={val_metrics['val_loss']:.4f} "
            f"| lr={lr_now:.2e} "
            f"| {epoch_time:.1f}s"
        )

        record = {"epoch": epoch, **train_metrics, **val_metrics, "lr": lr_now}
        history.append(record)

        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch:03d}.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            _print(f"  Saved checkpoint: {ckpt_path}")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_path = os.path.join(args.output_dir, "finetune_best.pth")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }, best_path)
            _print(f"  New best model (val_loss={best_val_loss:.4f}), saved: {best_path}")

    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    _print(f"\nTraining complete. History saved to {hist_path}")
    _print(f"Best checkpoint: {os.path.join(args.output_dir, 'finetune_best.pth')}")


if __name__ == "__main__":
    main()
