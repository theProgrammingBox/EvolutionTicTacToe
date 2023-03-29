#include <cudnn.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_fp16.h>
#include <iostream>

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

__global__ void GpuReluF16(__half* input, __half* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint16_t*)(input + index) >> 15)
		output[index] = 0x0000;
}

void ReluF16(__half* input, __half* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 1, cudaMemcpyDeviceToDevice);
	GpuReluF16 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}

int main()
{
	printf("cuDNN version: %d.%d.%d\n", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
	printf("cuBLAS version: %d.%d.%d\n", CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH);
	printf("cuRAND version: %d.%d.%d\n", CURAND_VERSION / 1000, (CURAND_VERSION % 1000) / 100, CURAND_VERSION % 100);
	printf("\n");

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);
	
	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

	const __half alphaf16 = __float2half(1.0f);
	const __half betaf16 = __float2half(0.0f);

	const uint32_t INPUT_SIZE = 10;
	const uint32_t HIDDEN_ONE_SIZE = 1;
	const uint32_t HIDDEN_TWO_SIZE = 1;
	const uint32_t OUTPUT_SIZE = 9;
	

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
	
	cudaMalloc(&gpuHiddenOneWeights, (INPUT_SIZE * HIDDEN_ONE_SIZE) << 1);
	cudaMalloc(&gpuHiddenTwoWeights, (HIDDEN_ONE_SIZE * HIDDEN_TWO_SIZE) << 1);
	cudaMalloc(&gpuOutputWeights, (HIDDEN_TWO_SIZE * OUTPUT_SIZE) << 1);
	cudaMalloc(&gpuHiddenOneBias, HIDDEN_ONE_SIZE << 1);
	

	CurandGenerateUniformF16(curandGenerator, gpuHiddenOneWeights, INPUT_SIZE * HIDDEN_ONE_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuHiddenTwoWeights, HIDDEN_ONE_SIZE * HIDDEN_TWO_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuOutputWeights, HIDDEN_TWO_SIZE * OUTPUT_SIZE);
	CurandGenerateUniformF16(curandGenerator, gpuHiddenOneBias, HIDDEN_ONE_SIZE);

	
	__half* cpuHiddenOneWeights = new __half[INPUT_SIZE * HIDDEN_ONE_SIZE];
	__half* cpuHiddenTwoWeights = new __half[HIDDEN_ONE_SIZE * HIDDEN_TWO_SIZE];
	__half* cpuOutputWeights = new __half[HIDDEN_TWO_SIZE * OUTPUT_SIZE];
	__half* cpuHiddenOneBias = new __half[HIDDEN_ONE_SIZE];

	
	cudaMemcpy(cpuHiddenOneWeights, gpuHiddenOneWeights, (INPUT_SIZE * HIDDEN_ONE_SIZE) << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuHiddenTwoWeights, gpuHiddenTwoWeights, (HIDDEN_ONE_SIZE * HIDDEN_TWO_SIZE) << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputWeights, gpuOutputWeights, (HIDDEN_TWO_SIZE * OUTPUT_SIZE) << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuHiddenOneBias, gpuHiddenOneBias, HIDDEN_ONE_SIZE << 1, cudaMemcpyDeviceToHost);

	
	PrintMatrixF16(cpuHiddenOneWeights, INPUT_SIZE, HIDDEN_ONE_SIZE, "Hidden One Weights");
	PrintMatrixF16(cpuHiddenTwoWeights, HIDDEN_ONE_SIZE, HIDDEN_TWO_SIZE, "Hidden Two Weights");
	PrintMatrixF16(cpuOutputWeights, HIDDEN_TWO_SIZE, OUTPUT_SIZE, "Output Weights");
	PrintMatrixF16(cpuHiddenOneBias, 1, HIDDEN_ONE_SIZE, "Hidden One Bias");

	
	float* board = new float[9];
	memset(board, 0, 9 * sizeof(float));
	float* isPlayerOne = new float[1];
	isPlayerOne[0] = 1.0f;

	__half* cpuInputMatrix = new __half[INPUT_SIZE];
	for (uint32_t i = 0; i < 9; i++)
		cpuInputMatrix[i] = __float2half(board[i]);
	cpuInputMatrix[9] = __float2half(isPlayerOne[0]);
	
	cudaMemcpy(gpuInputMatrix, cpuInputMatrix, INPUT_SIZE << 1, cudaMemcpyHostToDevice);
	
	__half* cpuCheckInputMatrix = new __half[INPUT_SIZE];
	cudaMemcpy(cpuCheckInputMatrix, gpuInputMatrix, INPUT_SIZE << 1, cudaMemcpyDeviceToHost);
	PrintMatrixF16(cpuCheckInputMatrix, 1, INPUT_SIZE, "Input Matrix");
	
	return 0;
}