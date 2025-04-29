# Settings for Google Colab

## Change Runtime for T4 Tesla (Colab GPU)
```
Runtime → Change runtime type → GPU
```

## Initialization
```bash
%%shell
python --version
nvcc --version
```
### Optional
```bash
pip install nvcc4jupyter
```

## Check GPU
```bash
%%shell
nvidia-smi
```

## Writefile Header
```bash
%%writefile hello_gpu.cu
```

## Compile & Run (Tesla T4)
```bash
%%shell
nvcc -arch=compute_75 -code=sm_75 hello_gpu.cu -o hello_gpu && ./hello_gpu
```

## Helpful in using Python CUDA in Colab

### Install numba-cuda
```bash
!uv pip install -q --system numba-cuda==0.4.0
```

### Specify the following in code
```python
from numba import config
config.CUDA_ENABLE_PYNVJITLINK = 1
```

# Python venv

## Create venv
```bash
python3 -m venv project-env
```

## Activate venv
```bash
source project-env/bin/activate
```

## Deactivate venv
```bash
deactivate
```