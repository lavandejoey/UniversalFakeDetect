#!/bin/bash
#SBATCH --job-name=UniFakeDetEval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=A40,L40S,A100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# -------- shell hygiene --------
# Exit immediately if a command exits with a non-zero status.
set -euo pipefail
# Enable debugging output
#set -x
umask 077
mkdir -p logs

# -------- print job header --------
echo "================= SLURM JOB START ================="
echo "Job:    $SLURM_JOB_NAME  (ID: $SLURM_JOB_ID)"
echo "Node:   ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-unknown}  (${SLURM_JOB_GPUS:-not-set})"
echo "Start:  $(date)"
echo "==================================================="

datetime="$(date '+%Y%m%d_%H%M%S')"
result_dir="results/${datetime}_clip_vitl14"
mkdir -p "${result_dir}"
#data_root="/home/infres/ziyliu-24/data/FakeParts2DataMock"
data_root="/projects/hi-paris/DeepFakeDataset/FakeParts_data_addition_frames_only"
arch="CLIP:ViT-L/14"
#'Imagenet:resnet18','Imagenet:resnet34','Imagenet:resnet50','Imagenet:resnet101','Imagenet:resnet152'
#'Imagenet:vgg11','Imagenet:vgg19',Imagenet:swin-b','Imagenet:swin-s','Imagenet:swin-t'
#'Imagenet:vit_b_16','Imagenet:vit_b_32','Imagenet:vit_l_16','Imagenet:vit_l_32'
#'CLIP:RN50', 'CLIP:RN101', 'CLIP:RN50x4', 'CLIP:RN50x16', 'CLIP:RN50x64'
#'CLIP:ViT-B/32', 'CLIP:ViT-B/16', 'CLIP:ViT-L/14', 'CLIP:ViT-L/14@336px'
data_entry_csv="/projects/hi-paris/DeepFakeDataset/frames_index.csv"
done_csv_list=("results")

source /home/infres/ziyliu-24/miniconda3/etc/profile.d/conda.sh
conda activate fakevlm310

srun python3 -Wignore UniFakeDetEval.py \
    --data_root "${data_root}" \
    --arch "${arch}" \
    --ckpt "pretrained_weights/fc_weights.pth" \
    --result_folder "${result_dir}" \
    --data_csv ${data_entry_csv} \
    --done_csv_list "${done_csv_list[@]}"

EXIT_CODE=$?

echo "================== SLURM JOB END =================="
echo "End:   $(date)"
echo "Exit:  ${EXIT_CODE}"
echo "==================================================="
exit "${EXIT_CODE}"