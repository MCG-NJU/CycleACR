# Installation

## Requirements

- Python 3.6 or higher

- [Pytorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision).<br>
  We can successfully reproduce the main results under the setting below:<br>
  Tesla A100 (80G): CUDA 11.1 + PyTorch 1.8.0 + torchvision 0.9.0

- [PyAV==8.0.3](https://github.com/PyAV-Org/PyAV)
- [yacs](https://github.com/rbgirshick/yacs)
- [OpenCV](https://opencv.org/)
- [tensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm)
- [FFmpeg](https://www.ffmpeg.org/)
- [Cython](https://cython.org/), [cython_bbox](https://github.com/samson-wang/cython_bbox), [SciPy](https://scipy.org/scipylib/), [matplotlib](https://matplotlib.org/), [easydict](https://github.com/makinacorpus/easydict) (for running demo)
- Linux + Nvidia GPUs

## Build

```
git clone https://github.com/MCG-NJU/CycleACR.git
cd CycleACR
pip install -e .  # Other dependicies will be installed here
python setup.py build develop
```

If have to reinstall, first:

```
rm -rf build
```

Now the installation is finished.

