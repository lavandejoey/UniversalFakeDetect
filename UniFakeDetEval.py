"""
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
import argparse
import logging
import numpy as np
import os
import pandas as pd
import pickle
import random
import shutil
import sys
import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy
from io import BytesIO
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from tqdm import tqdm

from DataUtils import standardise_predictions, index_dataframe
from models import get_model

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

SEED = 0


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

    return Image.fromarray(img)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def validate(model, loader):
    with torch.no_grad():
        y_true, y_score, idxs = [], [], []
        logger.info(f"Length of dataset: {len(loader)}")
        for img, label, idx in tqdm(loader, desc="Inference", leave=False):
            in_tens = img.cuda()
            # raw logits; DO NOT apply sigmoid
            y_score.extend(model(in_tens).flatten().tolist())
            y_true.extend(label.flatten().tolist())
            idxs.extend(idx.tolist())
    return np.array(y_true), np.array(y_score), np.array(idxs)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class FakePartsV2Dataset(Dataset):
    """
    A minimal dataset that reads from a dataframe (or list) of IndexEntry records:
      Columns/fields expected:
        root (Path or str), rel_path (Path or str), label (int in {0,1}),
        task (str), method (str), subset (str), mode (str: 'frame' or 'video')

    Notes:
    - We treat entries as image files; non-image rows are optionally skipped.
    - Robustness transforms (jpeg_quality / gaussian_sigma) mirror RealFakeDataset.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 arch: str,
                 jpeg_quality: int = None,
                 gaussian_sigma: int = None,
                 allow_modes=('frame',),
                 valid_exts=("png", "jpg", "jpeg", "bmp", "JPEG", "JPG"),
                 drop_missing=True):

        self.meta = df[['task', 'method', 'subset', 'label', 'mode', 'abs_path']].reset_index(drop=True)
        # Accept list of dataclass entries too
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([e.__dict__ for e in df])

        # normalise columns possibly as pathlib.Path where relevant
        def _to_path(x):
            from pathlib import Path
            return x if isinstance(x, Path) else Path(str(x))

        # Prefer abs_path if already provided by your indexer; else root/rel_path
        if 'abs_path' in df.columns:
            df['_abs_path'] = df['abs_path'].apply(_to_path)
        else:
            df['_abs_path'] = df.apply(lambda r: _to_path(r['root']) / _to_path(r['rel_path']), axis=1)

        # Filter by mode (default: only frames/images)
        if 'mode' in df.columns and allow_modes is not None:
            df = df[df['mode'].isin(allow_modes)].copy()

        # Filter by extension (quick check to avoid videos or odd files)
        df['_ext'] = df['_abs_path'].map(lambda p: p.suffix.lstrip('.'))
        df = df[df['_ext'].isin(valid_exts)].copy()

        # Drop rows whose files are missing (optional)
        if drop_missing:
            df = df[df['_abs_path'].map(lambda p: p.exists())].copy()

        # Minimal required fields
        assert 'label' in df.columns, "Dataframe must contain a 'label' column with 0/1."
        self.labels = df['label'].astype(int).to_numpy()
        self.paths = df['_abs_path'].map(lambda p: str(p)).to_list()

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = int(self.labels[idx])

        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma)
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label, idx


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    # parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_root', type=str, default=None, help='data root for both real and fake')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000,
                        help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None,
                        help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None,
                        help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

    args = parser.parse_args()

    os.makedirs(args.result_folder, exist_ok=True)

    model = get_model(args.arch)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    logger.info("Model loaded..")
    model.eval()
    model.cuda()

    # Columns: ['task','method','subset','label','mode','rel_path','abs_path','root']
    datasets: pd.DataFrame = index_dataframe(root_path=args.data_root)
    group_cols = [c for c in ['task', 'method'] if c in datasets.columns]
    if len(group_cols) == 0:
        group_cols = []
    groups = [("ALL", datasets)] if not group_cols else datasets.groupby(group_cols)

    for gkey, gdf in groups if group_cols else groups:
        # Human-friendly key
        key = gkey if isinstance(gkey, str) else "/".join(map(str, gkey))

        dataset = FakePartsV2Dataset(
            df=gdf if group_cols else gdf,  # gdf is df when no grouping
            arch=args.arch,
            jpeg_quality=args.jpeg_quality,
            gaussian_sigma=args.gaussian_sigma,
            allow_modes=('frame',)  # keep minimal & image-only by default
        )

        if len(dataset) == 0:
            logger.warning(f"[Skip] {key}: empty dataset after filtering.")
            continue

        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        # ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)
        all_rows = []
        y_true, y_score, idxs = validate(model, loader)
        meta = dataset.meta.iloc[idxs]  # aligns with returned idxs
        for j in range(len(idxs)):
            all_rows.append({
                "sample_id": str(meta.iloc[j]["abs_path"]),
                "task": str(meta.iloc[j]["task"]),
                "method": str(meta.iloc[j]["method"]),
                "subset": str(meta.iloc[j]["subset"]),
                "label": int(meta.iloc[j]["label"]),
                "model": f"UniFakeDet-{args.arch}-ckpt",
                "mode": str(meta.iloc[j]["mode"]),
                "score": float(y_score[j]),  # raw logit; no post-process
                "pred": int(y_score[j] > 0.0),  # hard label from logit
            })

        # after the for-loop over all groups:
        pred_df = standardise_predictions(all_rows)  # dtype hygiene + schema check
        save_path = os.path.join(args.result_folder, f"predictions_{key.replace('/', '_')}.csv")
        pred_df.to_csv(save_path, index=False)
        logger.info(f"[Done] {key}: {len(pred_df)} samples evaluated and saved.")
