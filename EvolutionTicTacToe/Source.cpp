#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

/*
IMPORTANT LESSONS
0. cudnnActivationForward needs a f32 alpha and beta when using f16 input and output
1. cudnnConvolutionForward needs a f32 alpha and beta when using f16 input and output
2. cublasGemmStridedBatchedEx needs a f16 alpha and beta when using f16 input and output
3. cublasAxpyEx needs a f32 alpha and beta when using f16 input and output
4. cudnnSoftmaxForward needs a f32 alpha and beta when using f16 input and output
*/

void PrintMatrixF16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
{
printf("%s:\n", label);
for (uint32_t i = 0; i < rows; i++)
{
	for (uint32_t j = 0; j < cols; j++)
		printf("%8.3f ", __half2float(arr[i * cols + j]));
	printf("\n");
}
printf("\n");
}

__global__ void CurandNormalizeF16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformF16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizeF16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.0000152590218967f);
}

int main()
{
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");

	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);
	cudnnHandle_t cudnnHandle;
	cudnnCreate(&cudnnHandle);
	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);
	
	const float alphaf32 = 1.0f;
	const float betaf32 = 0.0f;
	const __half alphaf16 = __float2half(1.0f);
	const __half betaf16 = __float2half(0.0f);

	const uint32_t INPUT_SIZE = 10;
	const uint32_t HIDDEN_ONE_SIZE = 16;
	const uint32_t HIDDEN_TWO_SIZE = 16;
	const uint32_t OUTPUT_SIZE = 9;
	const uint32_t HIDDEN_ONE_WEIGHTS_SIZE = HIDDEN_ONE_SIZE * INPUT_SIZE;
	const uint32_t HIDDEN_TWO_WEIGHTS_SIZE = HIDDEN_TWO_SIZE * HIDDEN_ONE_SIZE;
	const uint32_t OUTPUT_WEIGHTS_SIZE = OUTPUT_SIZE * HIDDEN_TWO_SIZE;
	
	__half* gpuInputMatrix;
	__half* gpuHiddenOneMatrix;
	__half* gpuHiddenTwoMatrix;
	__half* gpuOutputMatrix;
	__half* gpuHiddenOneWeights;
	__half* gpuHiddenTwoWeights;
	__half* gpuOutputWeights;
	__half* gpuHiddenOneBias;
	
	cudaMalloc(&gpuInputMatrix, INPUT_SIZE << 1);
	cudaMalloc(&gpuHiddenOneMatrix, HIDDEN_ONE_SIZE << 1);
	cudaMalloc(&gpuHiddenTwoMatrix, HIDDEN_TWO_SIZE << 1);
	cudaMalloc(&gpuOutputMatrix, OUTPUT_SIZE << 1);
	cudaMalloc(&gpuHiddenOneWeights, HIDDEN_ONE_WEIGHTS_SIZE << 1);
	cudaMalloc(&gpuHiddenTwoWeights, HIDDEN_TWO_WEIGHTS_SIZE << 1);
	cudaMalloc(&gpuOutputWeights, OUTPUT_WEIGHTS_SIZE << 1);
	cudaMalloc(&gpuHiddenOneBias, HIDDEN_ONE_SIZE << 1);
	
	CurandGenerateUniformF16(curandGenerator, gpuHiddenOneWeights, HIDDEN_ONE_WEIGHTS_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuHiddenTwoWeights, HIDDEN_TWO_WEIGHTS_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuOutputWeights, OUTPUT_WEIGHTS_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuHiddenOneBias, HIDDEN_ONE_SIZE);
	
	__half* cpuInputMatrix = new __half[INPUT_SIZE];
	__half* cpuHiddenOneMatrix = new __half[HIDDEN_ONE_SIZE];
	__half* cpuHiddenTwoMatrix = new __half[HIDDEN_TWO_SIZE];
	__half* cpuOutputMatrix = new __half[OUTPUT_SIZE];
	
	cudnnTensorDescriptor_t inputTensorDescriptor;
	cudnnTensorDescriptor_t hiddenOneTensorDescriptor;
	cudnnTensorDescriptor_t hiddenTwoTensorDescriptor;
	cudnnTensorDescriptor_t outputTensorDescriptor;

	cudnnCreateTensorDescriptor(&inputTensorDescriptor);
	cudnnCreateTensorDescriptor(&hiddenOneTensorDescriptor);
	cudnnCreateTensorDescriptor(&hiddenTwoTensorDescriptor);
	cudnnCreateTensorDescriptor(&outputTensorDescriptor);
	
	cudnnSetTensor4dDescriptor(inputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, INPUT_SIZE);
	cudnnSetTensor4dDescriptor(hiddenOneTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, HIDDEN_ONE_SIZE);
	cudnnSetTensor4dDescriptor(hiddenTwoTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, HIDDEN_TWO_SIZE);
	cudnnSetTensor4dDescriptor(outputTensorDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, OUTPUT_SIZE);
	
	cudnnActivationDescriptor_t reluActivationDescriptor;
	cudnnCreateActivationDescriptor(&reluActivationDescriptor);
	cudnnSetActivationDescriptor(reluActivationDescriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
	
	float* board = new float[9];
	float* isPlayerOne = new float;
	float* cpuActionProb = new float[9];
	memset(board, 0, 9 << 2);
	*isPlayerOne = 1.0f;

	for (uint32_t i = 9; i--;) {
		for (uint32_t i = 0; i < 9; i++)
			cpuInputMatrix[i] = __float2half(board[i]);
		cpuInputMatrix[9] = __float2half(*isPlayerOne);
		PrintMatrixF16(cpuInputMatrix, 1, INPUT_SIZE, "Input Matrix");

		cudaMemcpy(gpuInputMatrix, cpuInputMatrix, INPUT_SIZE << 1, cudaMemcpyHostToDevice);

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			HIDDEN_ONE_SIZE, 1, INPUT_SIZE,
			&alphaf16,
			gpuHiddenOneWeights, CUDA_R_16F, HIDDEN_ONE_SIZE, HIDDEN_ONE_WEIGHTS_SIZE,
			gpuInputMatrix, CUDA_R_16F, INPUT_SIZE, INPUT_SIZE,
			&betaf16,
			gpuHiddenOneMatrix, CUDA_R_16F, HIDDEN_ONE_SIZE, HIDDEN_ONE_SIZE,
			1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
		);

		cudaMemcpy(cpuHiddenOneMatrix, gpuHiddenOneMatrix, HIDDEN_ONE_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuHiddenOneMatrix, 1, HIDDEN_ONE_SIZE, "Hidden One Matrix");

		cublasAxpyEx
		(
			cublasHandle, HIDDEN_ONE_SIZE,
			&alphaf32, CUDA_R_32F,
			gpuHiddenOneBias, CUDA_R_16F, 1,
			gpuHiddenOneMatrix, CUDA_R_16F, 1,
			CUDA_R_32F
		);

		cudaMemcpy(cpuHiddenOneMatrix, gpuHiddenOneMatrix, HIDDEN_ONE_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuHiddenOneMatrix, 1, HIDDEN_ONE_SIZE, "Hidden One Matrix");

		cudnnActivationForward
		(
			cudnnHandle, reluActivationDescriptor,
			&alphaf32, hiddenOneTensorDescriptor, gpuHiddenOneMatrix,
			&betaf32, hiddenOneTensorDescriptor, gpuHiddenOneMatrix
		);

		cudaMemcpy(cpuHiddenOneMatrix, gpuHiddenOneMatrix, HIDDEN_ONE_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuHiddenOneMatrix, 1, HIDDEN_ONE_SIZE, "Hidden One Matrix");

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			HIDDEN_TWO_SIZE, 1, HIDDEN_ONE_SIZE,
			&alphaf16,
			gpuHiddenTwoWeights, CUDA_R_16F, HIDDEN_TWO_SIZE, HIDDEN_TWO_WEIGHTS_SIZE,
			gpuHiddenOneMatrix, CUDA_R_16F, HIDDEN_ONE_SIZE, HIDDEN_ONE_SIZE,
			&betaf16,
			gpuHiddenTwoMatrix, CUDA_R_16F, HIDDEN_TWO_SIZE, HIDDEN_TWO_SIZE,
			1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
		);

		cudaMemcpy(cpuHiddenTwoMatrix, gpuHiddenTwoMatrix, HIDDEN_TWO_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuHiddenTwoMatrix, 1, HIDDEN_TWO_SIZE, "Hidden Two Matrix");

		cudnnActivationForward
		(
			cudnnHandle, reluActivationDescriptor,
			&alphaf32, hiddenTwoTensorDescriptor, gpuHiddenTwoMatrix,
			&betaf32, hiddenTwoTensorDescriptor, gpuHiddenTwoMatrix
		);

		cudaMemcpy(cpuHiddenTwoMatrix, gpuHiddenTwoMatrix, HIDDEN_TWO_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuHiddenTwoMatrix, 1, HIDDEN_TWO_SIZE, "Hidden Two Matrix");

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			OUTPUT_SIZE, 1, HIDDEN_TWO_SIZE,
			&alphaf16,
			gpuOutputWeights, CUDA_R_16F, OUTPUT_SIZE, OUTPUT_WEIGHTS_SIZE,
			gpuHiddenTwoMatrix, CUDA_R_16F, HIDDEN_TWO_SIZE, HIDDEN_TWO_SIZE,
			&betaf16,
			gpuOutputMatrix, CUDA_R_16F, OUTPUT_SIZE, OUTPUT_SIZE,
			1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT
		);

		cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, OUTPUT_SIZE << 1, cudaMemcpyDeviceToHost);
		//PrintMatrixF16(cpuOutputMatrix, 1, OUTPUT_SIZE, "Output Matrix");

		cudnnSoftmaxForward
		(
			cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
			&alphaf32, outputTensorDescriptor, gpuOutputMatrix,
			&betaf32, outputTensorDescriptor, gpuOutputMatrix
		);

		cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, OUTPUT_SIZE << 1, cudaMemcpyDeviceToHost);
		PrintMatrixF16(cpuOutputMatrix, 1, OUTPUT_SIZE, "Output Matrix");

		for (int i = 0; i < OUTPUT_SIZE; i++)
		{
			cpuActionProb[i] = __half2float(cpuOutputMatrix[i]);
		}

		float randomNumber = (float)rand() / (float)RAND_MAX;
		uint32_t action = 0;
		for (;;)
		{
			randomNumber -= cpuActionProb[action];
			if (randomNumber <= 0.0f) break;
			action -= (++action >= OUTPUT_SIZE) * OUTPUT_SIZE;
		}

		board[action] = *isPlayerOne;
	}

	delete[] board;
	delete[] isPlayerOne;
	
	cudaFree(gpuInputMatrix);
	cudaFree(gpuHiddenOneMatrix);
	cudaFree(gpuHiddenTwoMatrix);
	cudaFree(gpuOutputMatrix);
	
	cudaFree(gpuHiddenOneWeights);
	cudaFree(gpuHiddenTwoWeights);
	cudaFree(gpuOutputWeights);
	cudaFree(gpuHiddenOneBias);
	
	delete[] cpuInputMatrix;
	delete[] cpuHiddenOneMatrix;
	delete[] cpuHiddenTwoMatrix;
	delete[] cpuOutputMatrix;
	
	cublasDestroy(cublasHandle);
	cudnnDestroy(cudnnHandle);
	curandDestroyGenerator(curandGenerator);
	
	cudnnDestroyTensorDescriptor(inputTensorDescriptor);
	cudnnDestroyTensorDescriptor(hiddenOneTensorDescriptor);
	cudnnDestroyTensorDescriptor(hiddenTwoTensorDescriptor);
	cudnnDestroyTensorDescriptor(outputTensorDescriptor);
	cudnnDestroyActivationDescriptor(reluActivationDescriptor);
	
	return 0;
}