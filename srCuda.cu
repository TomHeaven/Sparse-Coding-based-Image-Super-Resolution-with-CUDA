#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cmath>


//#define DEBUG

#ifdef DEBUG
#include "cuPrintf.cu"
#endif

/**
 * Device code
 */

#define PATCH_SIZE 5
#define C_HEIGHT 1024
#define DIM_FEA 100

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=false)
{
    if (code != cudaSuccess)
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      //  cudaDeviceReset();
        if (abort) exit(code);
    } else {
      #ifdef DEBUG
        printf("cuda returned code == cudaSuccess\n");
      #endif
    }
}


/**
 * @param hPatch - one-dim double array
 * @param mNorm - double
 * @param len - the length of hPatch
 *
 * Output altered hPatch
 */

__device__ void lin_scale(float* hPatch, const double mNorm, const int len) {
    // comput hNorm
    float hNorm = 0;
    for (int i = 0; i < len; ++i) {
        hNorm += hPatch[i] * hPatch[i];
    }
    hNorm = sqrt(hNorm);

    if (hNorm > 0.001) {
        float s = mNorm * 1.2 / hNorm;
        for (int i = 0; i < len; ++i) {
            hPatch[i] = hPatch[i] * s;
        }
    }
}

/**
 * % input:
 * % mIm, 低分辨率图像通过三次差值后得到的与高分辨率大小一致的图像，大小为[H,W]
 * % imFea, 低分辨率特征图像, 大小为[dim_fea, (H - patch_size + 1)*(W - patch_size + 1)] == [c_width, (H-patch_size+1)*(H-patch_size+1)]
 * % Dh, 高分辨率字典[patch_size ^ 2, 1024]                == [patch_size^2, c_height]
 * % Dl, 低分辨率字典[dim_fea, 1024]                       == [c_width,  c_height]
 * % children_sparse_coe, 基向量的稀疏系数,大小为[1024, dim_fea]  == [c_height, c_width]
 * % output:
 * %          hIm, 重建后高分辨率矩阵
 */
//(d_A, d_B, patch_size, N, m_width, d_coe, c_width, c_height, d_Dh, d_C, d_D);
void __global__  srKernel(const double* mIm, const double*imFea, const int patch_size, const int N, const int m_width, const double* children_sparse_coe, const int c_width, const int c_height, const double* Dh, float* hIm, int* cntMat, float* info)
{
    //int const idx = blockDim.x * blockIdx.x + threadIdx.x + startThreadNum;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * m_width + x;

    // mIm - m_width
    int W = m_width;
    int H = N / m_width;

    // boundry check
    if (y + patch_size + 2 > H || x + patch_size + 2 > W || x < 2 || y < 2)
        return;

    int f_width = (H - patch_size + 1)*(W - patch_size + 1);
    // Dh
    //int dh_height = patch_size * patch_size;
    int dh_width = c_height;
    // children_sparse_coe - c_height, c_width
    const int dim_fea = c_width;


    /////////////////////////////////////////////////////////////////////////////
    // 1. get mPatchFea

    //compute mfNorm
    float mPatch[PATCH_SIZE][PATCH_SIZE];
    float mPatchFea[DIM_FEA];
    int index_PatchFea = x*(H - patch_size + 1) + y; //(x - 1)*(H - patch_size + 1) + y;
    float mfNorm = 0;
    for (int i = 0; i < dim_fea; ++i) {
        mPatchFea[i] = imFea[i*f_width + index_PatchFea]; // [i][index_PatchFea] f_width mPatchFea
        mfNorm += mPatchFea[i] * mPatchFea[i];
    }
    mfNorm = sqrt(mfNorm);

    //compute mNorm
    float sumPatch = 0;
    for (int i = 0; i < patch_size; ++i) {
        for (int j = 0; j < patch_size; ++j) {
            int index = (i + y) *m_width + (j + x); //[i][j] m_width
            mPatch[i][j] = mIm[index];
            sumPatch += mIm[index];
        }
    }

    float mMean = sumPatch / (patch_size*patch_size);
    float mNorm = 0;
    for (int i = 0; i < patch_size; ++i) {
        for (int j = 0; j < patch_size; ++j) {
            int index = (i + y) *m_width + (j + x);
            mNorm += (mIm[index] - mMean)*(mIm[index] - mMean);
        }
    }
    mNorm = sqrt(mNorm);

    //compute mPatchFea
    if (mfNorm > 1) {
        for (int i = 0; i < dim_fea; ++i) {
            mPatchFea[i] /= mfNorm;
        }
    }

    /////////////////////////////////////////////////////////////////////////////
    // 2. compute w
    // mPatchFea == y, size patch * patch
    //double* y = mPatchFea; // y [dim_fea]

    float w[C_HEIGHT];
    for (int i = 0; i < c_height; ++i) {
        w[i] = 0;
        for (int j = 0; j < dim_fea; ++j) { // c_width == dim_fea
            w[i] += mPatchFea[j] * children_sparse_coe[i*c_width + j]; // [i][j] c_width
        }
    }


    ///////////////////////////////////////////////////////////////////////////////
    // 3. compute hPatch
    float hPatch[PATCH_SIZE*PATCH_SIZE]; // hPatch
    for (int i = 0; i < patch_size*patch_size; ++i) {
        hPatch[i] = 0;
        for (int k = 0; k < c_height; ++k) {
            hPatch[i] += Dh[i* dh_width + k] * w[k];  //Dh [i][k]  dh_width
        }
    }

    int len = patch_size*patch_size;
    lin_scale(hPatch, mNorm, len);
    for(int i = 0; i < len; ++i) {
        hPatch[i] += mMean;
    }

    ///////////////////////////////////////////////////////////////////////////////
    // 4. update hIm and cntMat
    int count = 0;
    for (int i = y; i < y + patch_size; ++i) {
        for (int j = x; j < x + patch_size; ++j) {
            atomicAdd(&hIm[i*m_width + j], hPatch[(j-x)*patch_size + (i-y)]); 
            atomicAdd(&cntMat[i*m_width + j], 1);
        }
    }

#ifdef DEBUG
    // if (x == 42 && y == 2) {
    if (idx == 1) {
        info[0] = index_PatchFea;
        info[1] = mfNorm;
        info[2] = mNorm;
        info[3] = mMean;
        info[4] =  y * m_width + x;
        info[5] = mIm[1002];
        info[6] = Dh[0];
        info[7] = hPatch[0];

        info[8] = x;
        info[9] = y;
        info[10] = hIm[y*m_width + x];

        for(int  i = 0;  i < 25; ++i) {
            info[12 + i] = mPatch[i / 5][i % 5];//mPatchFea[i]; //
        }

        for(int  i = 0;  i < 25; ++i) {
            info[40 + i] = hPatch[i]; //w[i];
        }
    }

    if (idx > info[39]) {
        // info[37] = x;
        // info[38] = y;
        info[39] = idx;
    }
#endif
}


/*
 * Host code
 */
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    mxGPUArray const *A;
    mxGPUArray const *B;
    mxGPUArray const *coe;
    mxGPUArray const *Dh;

    mxGPUArray *C;
    mxGPUArray *D;

    double const *d_A;
    double const *d_B;
    float *d_C;
    int *d_D;
    double * ptr;

    double const *d_coe;
    double const *d_Dh;
    int N;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";


    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs != 7) || !(mxIsGPUArray(prhs[0]))) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    A = mxGPUCreateFromMxArray(prhs[0]);
    B = mxGPUCreateFromMxArray(prhs[1]);
    ptr = mxGetPr(prhs[2]);
    int patch_size = int(ptr[0] + 0.5);
    ptr = mxGetPr(prhs[3]);
    int m_width = int(ptr[0] + 0.5);
    coe = mxGPUCreateFromMxArray(prhs[4]);
    ptr = mxGetPr(prhs[5]);
    int c_width = int(ptr[0] + 0.5);
    Dh = mxGPUCreateFromMxArray(prhs[6]);

#ifdef DEBUG
    printf("nrhs = %d, patch_size = %d, m_width = %d, c_width = %d\n", nrhs, patch_size, m_width, c_width);
#endif

#ifdef DEBUG
    printf("mxGPUGetClassID(A) = %d, mxDOUBLE_CLASS = %d\n", mxGPUGetClassID(A), mxDOUBLE_CLASS);
#endif

    /*
     * Verify that A really is a double array before extracting the pointer.
     */
    if (mxGPUGetClassID(A) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if (mxGPUGetClassID(B) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if (mxGPUGetClassID(coe) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }
    if (mxGPUGetClassID(Dh) != mxDOUBLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }


#ifdef DEBUG
    printf("break point 1\n");
#endif

    /*
     * Now that we have verified the data type, extract a pointer to the input
     * data on the device.
     */
    d_A = (double const *)(mxGPUGetDataReadOnly(A));
    d_B = (double const *)(mxGPUGetDataReadOnly(B));
    d_coe = (double const *)(mxGPUGetDataReadOnly(coe));
    d_Dh = (double const *)(mxGPUGetDataReadOnly(Dh));


    /* Create a GPUArray to hold the result and get its underlying pointer. */
    C = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
            mxGPUGetDimensions(A),
            mxSINGLE_CLASS,
            mxREAL, // == 0
            MX_GPU_INITIALIZE_VALUES);
    d_C = (float *)(mxGPUGetData(C));

    D = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(A),
            mxGPUGetDimensions(A),
            mxINT32_CLASS,
            mxREAL,
            MX_GPU_INITIALIZE_VALUES);
    d_D = (int *)(mxGPUGetData(D));

#ifdef DEBUG
    printf("break point 2\n");
#endif

    N = (int)(mxGPUGetNumberOfElements(A));

#ifdef DEBUG
    printf("break point 3\n");
    printf("N = %d\n", N);
#endif


    int coeN = (int)(mxGPUGetNumberOfElements(coe));
    int c_height = coeN / c_width; // == 1024  dictionary dims
    int dh_width = c_height;
    int dh_height = (int)(mxGPUGetNumberOfElements(Dh)) / dh_width; // dh_height == patch_size * patch_size

    float* d_info = NULL;
    float info[100];
    cudaMalloc((void **)&d_info, 100 *sizeof(float));
    cudaMemcpy(d_info, info, 100 *sizeof(float), cudaMemcpyHostToDevice);
#ifdef DEBUG
    printf("bpg = %d, tpb = %d\n", blocksPerGrid, threadsPerBlock);
    cudaPrintfInit();
#endif


    int TILE_WIDTH = 32;
    dim3 tpb(TILE_WIDTH , TILE_WIDTH );
    dim3 bpg(ceil(m_width / TILE_WIDTH ), ceil(N / m_width / TILE_WIDTH ));

    srKernel <<<bpg, tpb>>>(d_A, d_B, patch_size, N, m_width, d_coe, c_width, c_height, d_Dh, d_C, d_D, d_info);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

#ifdef DEBUG
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
    printf("break point 4\n");
    cudaMemcpy(info, d_info, 100 *sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 70; ++i)
        printf("info[%d] = %f\n", i, info[i]);
#endif
    cudaFree(d_info);
    /* Wrap the result up as a MATLAB gpuArray for return. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(C);
    plhs[1] = mxGPUCreateMxArrayOnGPU(D);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
    mxGPUDestroyGPUArray(C);
    mxGPUDestroyGPUArray(D);
    mxGPUDestroyGPUArray(coe);
    mxGPUDestroyGPUArray(Dh);
}
