// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32
#define HIST_SIZE 256

//@@ insert code here

__global__ void floatToChar(float *input, unsigned char *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int idx = y * width + x;
  
  if (x < width && y < height) {
    output[3*idx] = (unsigned char)(255 * input[3*idx]);
    output[3*idx+1] = (unsigned char)(255 * input[3*idx+1]);
    output[3*idx+2] = (unsigned char)(255 * input[3*idx+2]);
  }
}

__global__ void RGBToGrayScale(unsigned char *input, unsigned char *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int idx = y * width + x;
  unsigned char r,g,b;
  
  if (x < width && y < height) {
    r = input[3*idx];
    g = input[3*idx+1];
    b = input[3*idx+2];
    output[idx] = (unsigned char)(0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histogram(unsigned char *input, int *hist, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int idx = y * width + x; 
  
  if (x < width && y < height) {
    atomicAdd(&(hist[input[idx]]), 1);
  }
}

__global__ void scan(int *input, float *output, int wh) {
  output[0] = (1.0*input[0]) / wh;
  for (int i = 1; i < HIST_SIZE; i++) {
    output[i] = output[i-1] + (1.0*input[i]) / wh;
  }
}

__device__ float clamp(float x, float start, float end) {
  return min(max(x, start), end);
}

__global__ void equalization(unsigned char *input, unsigned char *output, float *cdf, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int idx = y * width + x; 
  int cdfmin = cdf[0];
  
  if (x < width && y < height) {
    output[3*idx] = clamp(255*(cdf[input[3*idx]] - cdfmin)/(1.0 - cdfmin), 0, 255.0);
    output[3*idx+1] = clamp(255*(cdf[input[3*idx+1]] - cdfmin)/(1.0 - cdfmin), 0, 255.0);
    output[3*idx+2] = clamp(255*(cdf[input[3*idx+2]] - cdfmin)/(1.0 - cdfmin), 0, 255.0);
  }
}

__global__ void charToFloat(unsigned char *input, float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; 
  int y = blockIdx.y * blockDim.y + threadIdx.y; 
  int idx = y * width + x;
  
  if (x < width && y < height) {
    output[3*idx] = (float)(input[3*idx]/255.0);
    output[3*idx+1] = (float)(input[3*idx+1]/255.0);
    output[3*idx+2] = (float)(input[3*idx+2]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *inputDataChar;
  unsigned char *grayImage;
  int *hist;
  float *cdf;
  unsigned char *outputDataChar;
  float *deviceOutputImageData;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  int imageSize = imageWidth * imageHeight * imageChannels;
  // Allocate GPU memory
  cudaMalloc((void **) &deviceInputImageData, imageSize * sizeof(float));
  cudaMalloc((void **) &inputDataChar, imageSize * sizeof(unsigned char));
  cudaMalloc((void **) &grayImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void **) &hist, HIST_SIZE * sizeof(int));
  cudaMalloc((void **) &cdf, HIST_SIZE * sizeof(float));
  cudaMalloc((void **) &outputDataChar, imageSize * sizeof(unsigned char));
  cudaMalloc((void **) &deviceOutputImageData, imageSize * sizeof(float));
  
  // Copy host memory to device
  cudaMemcpy(deviceInputImageData, hostInputImageData, imageSize*sizeof(float), cudaMemcpyHostToDevice);
  
  // Run kernel
  wbTime_start(Generic, "Calculation on kernel");
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 gridDim(ceil(1.0*imageWidth/BLOCK_SIZE), ceil(1.0*imageHeight/BLOCK_SIZE), 1);
  floatToChar<<<gridDim, blockDim>>>(deviceInputImageData, inputDataChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  RGBToGrayScale<<<gridDim, blockDim>>>(inputDataChar, grayImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  histogram<<<gridDim, blockDim>>>(grayImage, hist, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  scan<<<1,1>>>(hist, cdf, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  equalization<<<gridDim, blockDim>>>(inputDataChar, outputDataChar, cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  charToFloat<<<gridDim, blockDim>>>(outputDataChar, deviceOutputImageData, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbTime_stop(Generic, "Calculation on kernel");
  
  // Copy device memory to host
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, imageSize*sizeof(float), cudaMemcpyDeviceToHost);
  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  // Free GPU memor
  cudaFree(deviceInputImageData);
  cudaFree(inputDataChar);
  cudaFree(grayImage);
  cudaFree(hist);
  cudaFree(cdf);
  cudaFree(outputDataChar);
  cudaFree(deviceOutputImageData);
  
  // Free host memory
  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}

