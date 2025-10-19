#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x

export CUDA_VISIBLE_DEVICES=0
datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_clip_vitl14"
mkdir -p "${result_dir}"
data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
#data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
arch="CLIP:ViT-L/14"
#'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152'
#'Imagenet:vgg11','Imagenet:vgg19',Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t'
#'Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32'
#'CLIP:RN50', 'CLIP:RN101', 'CLIP:RN50x4', 'CLIP:RN50x16', 'CLIP:RN50x64'
#'CLIP:ViT-B/32', 'CLIP:ViT-B/16', 'CLIP:ViT-L/14', 'CLIP:ViT-L/14@336px'

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310

python3 UniFakeDetEval.py \
    --data_root "${data_root}" \
    --arch "${arch}" \
    --ckpt "pretrained_weights/fc_weights.pth" \
    --result_folder "${result_dir}"
