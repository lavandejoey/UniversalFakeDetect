"""
A40 ~ 1956MiB Model, 25204MiB Data (1024 Batch Size)

Usage:
conda activate fakevlm310

export CUDA_VISIBLE_DEVICES=0
datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_clip_vitl14"
mkdir -p "${result_dir}"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"

python3 UniFakeDetEval.py \
    --data_root "${data_root}" \
    --arch "CLIP:ViT-L/14" \
    --ckpt "pretrained_weights/fc_weights.pth" \
    --result_folder "${result_dir}"
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional

from DataUtils import (
    standardise_predictions,
    FakePartsV2DatasetBase,
    FRAMES_ROOT,
    FRAMES_CSV,
    collate_skip_none, find_best_threshold,
)
from models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073],
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711],
}


class FakePartsV2Dataset(FakePartsV2DatasetBase):
    def __init__(
            self,
            data_root: Union[str, Path] = FRAMES_ROOT,
            mode: str = "frame",  # "frame" or "video" (we expect "frame" here)
            csv_path: Optional[Union[str, Path]] = FRAMES_CSV,
            model_name: str = "unknown_model",
            arch: str = "CLIP:ViT-L/14",
            on_corrupt: str = "warn",  # safer default: keep going and skip bad files
            done_csv_list: Optional[Union[str, Path]] = None,
    ):
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        transform = T.Compose([
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

        super().__init__(
            data_root=data_root,
            mode=mode,
            csv_path=csv_path,
            model_name=model_name,
            transform=transform,
            on_corrupt=on_corrupt,
            done_csv_list=done_csv_list,
        )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_root", type=str, default=str(FRAMES_ROOT), help="dataset root folder")
    parser.add_argument("--data_mode", type=str, default="frame", choices=["frame", "video"], help="data modality")
    parser.add_argument("--data_csv", type=str, default=None, help="CSV indexing the dataset (optional)")
    parser.add_argument("--done_csv_list", type=str, nargs='*', default=[], help="List of done CSVs to skip samples")

    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default="./pretrained_weights/fc_weights.pth")

    parser.add_argument("--result_folder", type=str, default="result", help="folder to save predictions.csv")
    parser.add_argument("--batch_size", type=int, default=900)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))

    # parser.add_argument(
    #     "--jpeg_quality",
    #     type=int,
    #     default=None,
    #     help="JPEG quality in [1..100]; apply compression if set",
    # )
    # parser.add_argument(
    #     "--gaussian_sigma",
    #     type=int,
    #     default=None,
    #     help="Gaussian blur sigma (e.g., 1..4); apply blur if set",
    # )

    args = parser.parse_args()
    set_seed()

    os.makedirs(args.result_folder, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    model = get_model(args.arch)

    # PyTorch 2.6 default: weights_only=True; force legacy behaviour for broad compatibility
    try:
        state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(args.ckpt, map_location="cpu")

    # If checkpoint wrapped like {"state_dict": ...}
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.fc.load_state_dict(state_dict, strict=False)
    log.info("Checkpoint loaded into model.fc from: %s", args.ckpt)
    model.eval().to(device)

    dataset = FakePartsV2Dataset(
        data_root=args.data_root,
        mode=args.data_mode or "frame",
        csv_path=args.data_csv or FRAMES_CSV,
        model_name=f"UniFakeDet-{args.arch}",
        arch=args.arch,
        done_csv_list=args.done_csv_list,
    )
    log.info(f"Dataset: {dataset}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_skip_none,
        persistent_workers=(args.workers or 0) > 0,
        drop_last=False,
    )

    rows = []
    seen = 0
    for batch in tqdm(loader, desc="Inference", leave=False):
        if batch is None:
            # Entire batch got skipped due to corrupt files
            log.warning("All samples in batch skipped due to file errors.")
            continue

        imgs, labels, metas = batch  # imgs [B,3,224,224], labels [B], metas: dict[list]
        B = imgs.shape[0]
        seen += B

        imgs = imgs.to(device, non_blocking=True)

        with torch.no_grad():
            logits = model(imgs)

        # Flexible handling of output head
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)  # [B,1]

        if logits.shape[1] == 2:
            probs = F.softmax(logits.float(), dim=1)[:, 1]  # probability of "fake"
            preds = torch.argmax(logits, dim=1)
        elif logits.shape[1] == 1:
            probs = torch.sigmoid(logits.float()).squeeze(1)
            preds = (probs >= 0.5).long()
        else:
            raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

        probs_np = probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # metas is dict of lists aligned with batch dimension
        # Required schema columns come from DataUtils.REQUIRED_COLS
        for i in range(B):
            rows.append(
                {
                    "sample_id": str(metas["sample_id"][i]),
                    "task": str(metas["task"][i]),
                    "method": str(metas["method"][i]),
                    "subset": str(metas["subset"][i]),
                    "label": int(labels_np[i]),
                    "model": str(metas["model"][i]),
                    "mode": str(metas["mode"][i]),
                    "score": float(probs_np[i]),  # probability of fake
                    "pred": int(preds_np[i]),  # hard prediction
                }
            )

    log.info(f"Done inference: {seen:d} samples processed.")

    pred_df = standardise_predictions(rows)  # dtype hygiene + schema check
    # Update df by find_best_threshold
    best_th = find_best_threshold(pred_df["score"].values, pred_df["label"].values)
    log.info(f"Best threshold found: {best_th:.4f}", )
    pred_df["pred"] = (pred_df["score"] >= best_th).astype(int)
    save_path = os.path.join(args.result_folder, "predictions.csv")
    pred_df.to_csv(save_path, index=False)
    log.info(f"Saved predictions to {save_path} (rows={len(pred_df):d})")


if __name__ == "__main__":
    main()
