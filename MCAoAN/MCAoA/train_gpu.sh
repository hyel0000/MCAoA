#!/bin/bash
#SBATCH --job-name=train      # 잡 이름
#SBATCH --partition=suma_a6000      
#SBATCH --gres=gpu:A6000:1           
#SBATCH --time=24:00:00                 # 최대 실행 시간
#SBATCH --output=logs/%j.out            # Slurm 로그 저장 (%j = job ID)

# 1) Conda 환경 활성화
source ~/anaconda3/bin/activate hlkim

# 2) PyTorch가 자기 cuDNN 라이브러리를 우선 쓰도록 LD_LIBRARY_PATH 수정
export LD_LIBRARY_PATH=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")

# 3) GPU/환경 정보 출력 (디버깅용)
echo "=== Slurm Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
nvidia-smi
echo ""
python - <<'EOF'
import torch
print("PyTorch:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
EOF
echo ""

# 4) 학습 실행
python -u run.py --RUN='train' --MODEL='small' --SPLIT='train' --EVAL_EE=True --VERSION='small_train_val' --VERB=True
