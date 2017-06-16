# Sparse Coding based Image Super-resolution with CUDA

By Hanlin Tan, NUDT


This project implements the image super resolution algorithm of the paper ``
LASSO Approximation and Application to Image Super-resolution with CUDA Acceleration`` using matlab.

This project benefits from Yang's paper ``Image super-resolution via sparse representation`` and code.


## Hardware Requirements
You need a Nvidia GPU with at least 3GB RAM to run CUDA version of the proposed image super-resolution algorithm.

## Software Requirements
You need to intall:

+ matlab 2014a or later.
+ CUDA driver.
+ CUDA 8.0 SDK. Update ``mex_CUDA_*.xml`` if you use a different version.
+ Visual Studio 2013, for Windows OS. Update ``mex_CUDA_win64.xml`` if you use a different version.
+ XCode 8.0, for Mac OS X. Update ``mex_CUDA_maci64.xml`` if you use a different version.
+ GCC, for Linux OS.

## Compile mex CUDA program
You can use pre-compiled mex files, or compile ``srCuda.cu`` using matlab command

```
mex -v srCuda.cu
```
The above command will generate 

+ ``srCuda.mexw64`` on Windows
+ ``srCuda.mexmaci64`` on Mac OS X
+ ``srCuda.mexa64`` on Linux

using compiler configurations stored in ``mex_CUDA_win64.xml``, ``mex_CUDA_maci64.xml`` and ``mex_CUDA_glnxa64.xml``,respectively.

Note if you install different versions of CUDA or native C++ compiler, you need to update the xml files with correct version information.
### Notes for Linux
On Linux, you may need to install ``Matlab 2017a`` to run the code with ``CUDA 8.0``. Otherwise, you may encounter an error:
```
Can't reload '/usr/local/MATLAB/R2016a/bin/glnxa64/libmwgpu.so'
```

Another choice is to install ``CUDA 7.5`` and ``Matlab 2016a``.



## Replicate the results in the paper
To replicate the results, including figures, tables and result images in our paper:

+ To evaluate comparable algorithms, run ``compareAlgorithms.m``.
+ To replicate the figure and tables in the paper, run ``analyze.m``.

## Train your own dictionaries

The algorithms use a pair of sparse dictionaries in ``Dictionary`` folder. You can train your own dictionaries by cd to the train folder and run ``Demo_Dictionary_Training.m``. 

Our algorithm also relies on pre-computed sparse coefficents of base vectors stored in ``Children_sparse_coe.mat``, which can be updated by calling ``Children_SC(D)``, where ``D`` is the trained low resolution dictionary.

## Reference
If you use the code provided by the project, please cite our paper:

```
@inproceedings{tan2017lasso,
  title={LASSO Approximation and Application to Image Super-resolution with CUDA Acceleration},
  author={Tan, Halin and Zeng, Xiangrong and Lai, Shiming and Liu, Yu and Zhang, Maojun},
  booktitle={Image, Vision and Computing (ICIVC), International Conference on},
  year={2017},
  organization={IEEE}
}
``` 