https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements

The goal is 
GPU VM 
NVIDIA-SMI version  : 565.57.01
NVML version        : 565.57
DRIVER version      : 565.57.01
CUDA Version        : 12.7

based on the link up, so we want run onnxruntime-gpu that 1.20.x and PyTorch >= 2.4.0

here we choose 2.4.1 based on this link: https://github.com/pytorch/vision#installation
we need torchvision=0.19

we find docker image on this link
https://github.com/cnstark/pytorch-docker/tree/main



Package            Version
------------------ ------------
certifi            2023.7.22
charset-normalizer 3.2.0
cmake              3.27.4.1
filelock           3.12.3
idna               3.4
Jinja2             3.1.2
lit                16.0.6
MarkupSafe         2.1.3
mpmath             1.3.0
networkx           3.1
numpy              1.25.2
Pillow             10.0.0
pip                23.2.1
requests           2.31.0
setuptools         65.5.0
sympy              1.12
torch              2.0.1+cu118
torchaudio         2.0.2+cu118
torchvision        0.15.2+cu118
triton             2.0.0
typing_extensions  4.7.1
urllib3            2.0.4