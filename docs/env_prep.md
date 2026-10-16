# Prepare the environment

Use a Linux machine with an NVIDIA GPU for video evaluation and training. From the repository root:

```bash
conda create -n malow-video python=3.11 -y
conda activate malow-video
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-video.txt
python -m pip install --no-deps -e .
# Ubuntu/Debian; provides the ffmpeg and ffprobe executables for preparation.
sudo apt-get install ffmpeg
```

For the CUDA acceleration packages used in the reference environment, install these after PyTorch. Building requires a compatible local CUDA toolkit and compiler:

```bash
python -m pip install ninja packaging wheel setuptools
MAX_JOBS=8 python -m pip install --no-build-isolation flash-attn==2.8.3
python -m pip install --no-deps fla-core==0.5.0 flash-linear-attention==0.5.0
CAUSAL_CONV1D_FORCE_BUILD=TRUE python -m pip install --no-deps --no-build-isolation causal-conv1d==1.6.2.post1
```

These kernels accelerate Qwen3.5's attention and recurrent operations. CPU import/unit tests do not require them.