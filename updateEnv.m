% set env

p = getenv('PATH');
p = strcat(p, ':/Developer/NVIDIA/CUDA-7.5/bin');
setenv('PATH', p);
setenv('MW_NVCC_PATH', '/Developer/NVIDIA/CUDA-7.5/bin');
setenv('MW_XCODE_CLANG_COMPILER', '/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang');