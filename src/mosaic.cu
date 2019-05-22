/*
* This is the assignment 1 program code
* Student name is Tao Wu
* Email is twu17@sheffield.ac.uk
* Student Register No. is 180127601
* I declare that this code is my own work
*/

//********************The .exe file under x64\Debug folder ******************//

//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define FAILURE 0
#define SUCCESS !FAILURE
#define THREADS_PRE_BLOCK_X_32 32
#define THREADS_PRE_BLOCK_Y_32 32

#define USER_NAME "acp18tw"		//replace with your user name

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef enum PPM_FORMAT { PPM_BINARY, PPM_PLAIN_TEXT } PPM_FORMAT;

struct PPMrgb{
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

void print_help();
void checkCUDAError(const char *msg);
int process_command_line(int argc, char *argv[]);

// self function declaretion
unsigned int checkPowerOfTwo(unsigned int x);
struct PPMrgb* readPPMFile(char *filename, unsigned int *w, unsigned int *h, unsigned int *d);
struct PPMrgb average_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);
struct PPMrgb average_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);
int writePPMFile(char *filename, struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, PPM_FORMAT format);
int mosaicCal_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize);
int mosaicCal_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize);
int cudaMosaic_cal();
void freeAll(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);

// globe variables declaretion
unsigned int c = 0;
char *input_fileName, *output_fileName;
MODE execution_mode = CPU;
PPM_FORMAT ppm_format = PPM_BINARY;

// function of cuda implementation
__global__ void mosaicCal_cuda(float *inValue, float *ouValue, float *w, float *h, float *mSize)
{
  // share memory of image
  extern __shared__ float buffer[];
  // share memory of mosaic size in x and y direction
  __shared__ float blockSize_x;
  __shared__ float blockSize_y;
  __shared__ int width_int, height_int, mosaic_size;
  width_int = (int) (*w);
  height_int = (int) (*h);
  mosaic_size = (int) (*mSize);

  int i,j;
  int rgbIndex;
  // calculate position index in block;
  int index = blockDim.x * threadIdx.y + threadIdx.x;
  int w_index = blockIdx.x * mosaic_size + threadIdx.x;
  int h_index = blockIdx.y * mosaic_size + threadIdx .y;

  // initialize burffer
  buffer[index] = 0;
  if (index == 0) {
    blockSize_x = 0.0;
    blockSize_y = 0.0;
  }

  // calculate number pixels in a filter/block
  if (blockIdx.x == (gridDim.x - 1) && width_int % mosaic_size != 0)
    blockSize_x = (float)(width_int % mosaic_size);
  else
    blockSize_x = mosaic_size;
  if (blockIdx.y == (gridDim.y - 1) && height_int % mosaic_size != 0)
    blockSize_y = (float)(height_int % mosaic_size);
  else
    blockSize_y = mosaic_size;

  __syncthreads();

  // allocate the pixels to buffer
  //rgbIndex = h_index * width_int + w_index;
  for (j = threadIdx.y; j < blockSize_y; j += blockDim.y) {
    for(i = threadIdx.x; i < blockSize_x; i += blockDim.x) {
      rgbIndex = (blockIdx.y * mosaic_size + j) * width_int + w_index;
      buffer[index] += inValue[rgbIndex];
      rgbIndex += blockDim.x;
    }
    //rgbIndex = (blockIdx.y * mosaic_size + j) * width_int + w_index;
  }
  __syncthreads();

  // get total sum in buffer[0]
  for (unsigned int stride = blockDim.x*blockDim.y/2; stride > 0; stride >>= 1) {
    if (index < stride){
      buffer[index] += buffer[index + stride];
    }
    __syncthreads();
  }
  __syncthreads();

  // average the sum at buffer[0]
  if (index == 0) {
    buffer[0] = buffer[0] / (blockSize_x * blockSize_y);
  }
  __syncthreads();

  // resize to original size
  for (j = threadIdx.y; j < blockSize_y; j += blockDim.y) {
    for(i = threadIdx.x; i < blockSize_x; i += blockDim.x) {
      rgbIndex = (blockIdx.y * mosaic_size + j) * width_int + w_index;
      ouValue[rgbIndex] = buffer[0];
      rgbIndex += blockDim.x;
    }
  }
}

// function of calculating average in cuda
__global__ void average_cuda(float *inpt, float *oupt, float *w, float *h)
{
  extern __shared__ float tmp[];
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  // insert to shared memory
  tmp[threadIdx.x] = inpt[idx];

  __syncthreads();

  // calculate average
  for (unsigned int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride){
      tmp[threadIdx.x] += tmp[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // calculate block sum, add array[0] value to output
  if (threadIdx.x == 0) {
    atomicAdd(oupt, tmp[0]);
  }
}






int main(int argc, char *argv[]) {

	// declear variables;
  clock_t begin_single, end_single;
  double begin_multiple, end_multiple;
  double seconds_single, seconds_multiple;
  unsigned int *width, *height, *deepth;
	struct PPMrgb *ppmrgb;
	struct PPMrgb *ppmrgb_use_twice;
  struct PPMrgb averageRGB_single;
  struct PPMrgb averageRGB_multiple;
	// alocate memory for pointer
  width  = (unsigned int*)malloc(sizeof(unsigned int));
  height = (unsigned int*)malloc(sizeof(unsigned int));
  deepth = (unsigned int*)malloc(sizeof(unsigned int));

	if (process_command_line(argc, argv) == FAILURE)
		return 1;

	//TODO: read input image file (either binary or plain text PPM)
  ppmrgb = readPPMFile(input_fileName, width, height, deepth);
  // the RGB stored value changes after mosaic calculation.
  // it need read again, if execution_mode is ALL
  if (execution_mode == ALL)
    ppmrgb_use_twice = readPPMFile(input_fileName, width, height, deepth);

	//TODO: execute the mosaic filter based on the mode
	switch (execution_mode){
		case (CPU) : {
			//TODO: starting timing here
  		begin_single = clock();

			//TODO: calculate the average colour value
      averageRGB_single = average_single(ppmrgb, width, height, deepth);
  		if (mosaicCal_single(ppmrgb, width, height, deepth, c) == FAILURE)
    		return 1;
			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d\n", averageRGB_single.red, averageRGB_single.green, averageRGB_single.blue);

			//TODO: end timing here
  		end_single = clock();
      seconds_single = (double)(end_single - begin_single) / CLOCKS_PER_SEC;
			printf("CPU mode execution time took %.2fs and %.2fms\n",seconds_single, seconds_single*1000);
	    //save the output image file (from last executed mode)
      if (writePPMFile(output_fileName, ppmrgb, width, height, deepth, ppm_format) == FAILURE)
        return 1;
			break;
		}
		case (OPENMP) : {
			//TODO: starting timing here
      begin_multiple = omp_get_wtime();

			//TODO: calculate the average colour value
      averageRGB_multiple = average_multiple(ppmrgb, width, height, deepth);
  		if (mosaicCal_multiple(ppmrgb, width, height, deepth, c) == FAILURE)
    		return 1;

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d, green = %d, blue = %d\n", averageRGB_multiple.red, averageRGB_multiple.green, averageRGB_multiple.blue);

			//TODO: end timing here
      end_multiple = omp_get_wtime();
      seconds_multiple = (double)(end_multiple - begin_multiple);
			printf("OPENMP mode execution time took %.2fs and %.2fms\n", seconds_multiple, seconds_multiple*1000);
	    //save the output image file (from last executed mode)
      if (writePPMFile(output_fileName, ppmrgb, width, height, deepth, ppm_format) == FAILURE)
        return 1;
			break;
		}
		case (CUDA) : {
      // cuda calculation
      if(cudaMosaic_cal() == FAILURE)
        return 1;
			break;
		}
		case (ALL) : {
      // CPU implementation
			//TODO: starting timing here
  		begin_single = clock();

			//TODO: calculate the average colour value
      averageRGB_single = average_single(ppmrgb, width, height, deepth);
  		if (mosaicCal_single(ppmrgb, width, height, deepth, c) == FAILURE)
    		return 1;
			// Output the average colour value for the image
			printf("CPU Average image colour red = %d, green = %d, blue = %d\n", averageRGB_single.red, averageRGB_single.green, averageRGB_single.blue);

			//TODO: end timing here
  		end_single = clock();
      seconds_single = (double)(end_single - begin_single) / CLOCKS_PER_SEC;
			printf("CPU mode execution time took %.2fs and %.2fms\n",seconds_single, seconds_single*1000);

      printf("\n");
      // OPENMP implementation
			//TODO: starting timing here
      begin_multiple = omp_get_wtime();

			//TODO: calculate the average colour value
      averageRGB_multiple = average_multiple(ppmrgb_use_twice, width, height, deepth);
  		if (mosaicCal_multiple(ppmrgb_use_twice, width, height, deepth, c) == FAILURE)
    		return 1;

			// Output the average colour value for the image
			printf("OPENMP Average image colour red = %d, green = %d, blue = %d\n", averageRGB_multiple.red, averageRGB_multiple.green, averageRGB_multiple.blue);

			//TODO: end timing here
      end_multiple = omp_get_wtime();
      seconds_multiple = (double)(end_multiple - begin_multiple);
			printf("OPENMP mode execution time took %.2fs and %.2fms\n", seconds_multiple, seconds_multiple*1000);

      // cuda implementation
      if (cudaMosaic_cal() == FAILURE)
        return 1;
			break;
		}
	}


  // free memory
  if (execution_mode == ALL)
    free(ppmrgb_use_twice);
  freeAll(ppmrgb, width, height, deepth);
	free(input_fileName);
	free(output_fileName);
	return 0;
}

void print_help(){
	printf("mosaic_%s C M -i input_file -o output_file [options]\n", USER_NAME);

	printf("where:\n");
	printf("\tC              Is the mosaic cell size which should be any positive\n"
		   "\t               power of 2 number \n");
	printf("\tM              Is the mode with a value of either CPU, OPENMP, CUDA or\n"
		   "\t               ALL. The mode specifies which version of the simulation\n"
		   "\t               code should execute. ALL should execute each mode in\n"
		   "\t               turn.\n");
	printf("\t-i input_file  Specifies an input image file\n");
	printf("\t-o output_file Specifies an output image file which will be used\n"
		   "\t               to write the mosaic image\n");
	printf("[options]:\n");
	printf("\t-f ppm_format  PPM image output format either PPM_BINARY (default) or \n"
		   "\t               PPM_PLAIN_TEXT\n ");
}

int process_command_line(int argc, char *argv[]){
	unsigned int check_c;
  int i;

	if (argc < 7){
		fprintf(stderr, "Error: Missing program arguments. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//first argument is always the executable name
	if (strstr(argv[0], "mosaic_acp18tw") == NULL && strstr(argv[0], "mosaic") == NULL) {
		fprintf(stderr, "Error: error executable name. Re-complie and name mosaic_acp18tw or mosaic.\n");
		fprintf(stderr, "Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	//read in the non optional command line arguments
	c = (unsigned int)atoi(argv[1]);
	if (c < 0) {
		fprintf(stderr, "Error: mosaic size should larger than 0\n");
		print_help();
		return FAILURE;
	}
	check_c = checkPowerOfTwo(c);
	if (check_c == 2 || check_c == 0){
		fprintf(stderr, "Error: mosaic size not power of 2 number. Correct usage is...");
		print_help();
		return FAILURE;
	}

	//TODO: read in the mode
	if (strcmp(argv[2], "CPU") == 0)
		execution_mode = CPU;
	else if (strcmp(argv[2], "OPENMP") == 0)
		execution_mode = OPENMP;
	else if (strcmp(argv[2], "CUDA") == 0)
		execution_mode = CUDA;
	else if (strcmp(argv[2], "ALL") == 0)
		execution_mode = ALL;
	else if (strcmp(argv[2], "cpu") == 0)
		execution_mode = CPU;
	else if (strcmp(argv[2], "openmp") == 0)
		execution_mode = OPENMP;
	else if (strcmp(argv[2], "cuda") == 0)
		execution_mode = CUDA;
	else if (strcmp(argv[2], "all") == 0)
		execution_mode = ALL;
	else {
		fprintf(stderr, "Error: check entered mode. (e.g. CPU or cpu, not support to cPU like this) Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//TODO: read in the input image name
	if (strcmp(argv[3], "-i") != 0){
		fprintf(stderr, "Error: unknow argument. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	int input_fileCount = strlen(argv[4])+1;
	input_fileName = (char *) malloc(input_fileCount * sizeof(char));
	strcpy(input_fileName, argv[4]);
	// check if file exists
  FILE *testFile;
  if (testFile = fopen(input_fileName, "r")){
    fclose(testFile);
	}else{
    fclose(testFile);
		fprintf(stderr, "Error: intput file not exist");
		return FAILURE;
  }

	//TODO: read in the output image name
	if (strcmp(argv[5], "-o") != 0){
		fprintf(stderr, "Error: unknow argument. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	int output_fileCount = strlen(argv[6])+1;
	output_fileName = (char *) malloc(output_fileCount * sizeof(char));
	strcpy(output_fileName, argv[6]);
	// check input filename exist.
  //if (access(output_fileName, 0) != -1){
		//fprintf(stderr, "Error: output file exists");
		//return FAILURE;
	//}

	//TODO: read in any optional part 3 arguments
  i = 7;
  while(argv[i] != NULL){
	if (strcmp(argv[7], "-f") == 0){
		  if (strcmp(argv[8], "PPM_BINARY") == 0)
			   ppm_format = PPM_BINARY;
		  else if (strcmp(argv[8], "PPM_PLAIN_TEXT") == 0)
			   ppm_format = PPM_PLAIN_TEXT;
		  if (strcmp(argv[8], "ppm_binary") == 0)
			   ppm_format = PPM_BINARY;
		  else if (strcmp(argv[8], "ppm_plain_text") == 0)
			   ppm_format = PPM_PLAIN_TEXT;
      else {
			   fprintf(stderr, "Error: unvalid ppm_format. Correct usage is...\n");
			   print_help();
			   return FAILURE;
      }
		}
    i++;
	}

	return SUCCESS;
}

// check if a number is power of 2, return 1
// if a number is 0, return 0
// if a number is not power of 2, return 2
unsigned int checkPowerOfTwo(unsigned int x)
{
	if (x == 0)
		return 0;

	while (x != 1)
	{
		if (x % 2 != 0)
			return 2;
		x /= 2;
	}
	return 1;
}

// read PPM file, whatever is P3 or P6
struct PPMrgb* readPPMFile(char *filename, unsigned int *w, unsigned int *h, unsigned int *d)
{
  int line = 0;
  unsigned int getW, getH, getD;
  int i,j;
  char header[4][100];
  struct PPMrgb *rgb;
  FILE *f = NULL;
  f = fopen(filename, "r");

  // get first 4 parameters
  while(line < 4){
    fgets(header[line], 100, f);
    if (strncmp(header[line], "#", 1) != 0)
      ++line;
  }

  // get or check first 4 parameters
  getW = (unsigned int)strtol(header[1], NULL, 10);
  getH = (unsigned int)strtol(header[2], NULL, 10);
  getD = (unsigned int)strtol(header[3], NULL, 10);
  *w = getW;
  *h = getH;
  *d = getD;
  rgb = (struct PPMrgb*)malloc(sizeof(struct PPMrgb)* (*w) * (*h));
  if (strncmp(header[0], "P3", 2) == 0) {
    for (i = 0; i < *h; i++){
      for (j = 0; j < *w; j++) {
        if (j == (*w) -1)
          fscanf(f, "%hhu %hhu %hhu\n", &rgb[i*(*w)+j].red, &rgb[i*(*w)+j].green, &rgb[i*(*w)+j].blue);
        else
          fscanf(f, "%hhu %hhu %hhu\t", &rgb[i*(*w)+j].red, &rgb[i*(*w)+j].green, &rgb[i*(*w)+j].blue);
      }
    }
    fclose(f);
    return rgb;
  } else if (strncmp(header[0], "P6", 2) == 0) {
      fread(rgb, sizeof(struct PPMrgb), (*w) * (*h), f);
      fclose(f);
      return rgb;
  } else {
		fprintf(stderr, "Read Error: Support P3 and P6 only\n");
    fclose(f);
    freeAll(rgb, w, h, d);
    return FAILURE;
  }
}

// write to .ppm file with ppm format
int writePPMFile(char *filename, struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, PPM_FORMAT format)
{
  int i,j;
  FILE *f;
  f = fopen(filename, "w");
  // write first 4 parameters
  if (format == PPM_PLAIN_TEXT)
    fprintf(f, "P3\n");
  else if (format == PPM_BINARY)
    fprintf(f, "P6\n");
  else {
    fprintf(stderr, "Write Error: Support P3 and P6 only\n");
    freeAll(rgb, w, h, d);
    return FAILURE;
  }
  fprintf(f, "#COM4521 Assignment\n");
  fprintf(f, "%d\n", *w);
  fprintf(f, "%d\n", *h);
  fprintf(f, "%d\n", *d);

  // write P3 and P6 PPM rgb
  if (format == PPM_PLAIN_TEXT) {
    for (i = 0; i < *h; i++){
      for (j = 0; j < *w; j++) {
        if (j == (*w) - 1)
          fprintf(f, "%hhu %hhu %hhu\n", rgb[i*(*w)+j].red, rgb[i*(*w)+j].green, rgb[i*(*w)+j].blue);
        else
          fprintf(f, "%hhu %hhu %hhu\t", rgb[i*(*w)+j].red, rgb[i*(*w)+j].green, rgb[i*(*w)+j].blue);
      }
    }
    fclose(f);
    return SUCCESS;
  } else if (format == PPM_BINARY) {
      fwrite(rgb, sizeof(struct PPMrgb), (*w) * (*h), f);
      fclose(f);
      return SUCCESS;
  } else {
		fprintf(stderr, "Save Error: Support P3 and P6 only\n");
    fclose(f);
    freeAll(rgb, w, h, d);
    return FAILURE;
  }
}

// mosaic calculation for single thread (CPU)
int mosaicCal_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize)
{
  int i, j, buffer_i, buffer_j, rgbIndex;
  int cellW, cellH, particle_cellH, particle_cellW;
  int *counter;
  float *bufferRed, *bufferGreen, *bufferBlue;
  // Calculation of masic filter
  cellW = (*w) / (int)cSize;
  cellH = (*h) / (int)cSize;
  particle_cellW = (*w) % (int)cSize;
  particle_cellH = (*h) % (int)cSize;
  if (cSize < 0 || cSize >= (*w) || cSize >= (*h)) {
		fprintf(stderr, "Calculation Error: masic filter size is too large\n");
    return FAILURE;
  }
  // consider having particle mosaic cell
  if (particle_cellW != 0 || particle_cellH != 0) {
    cellW += 1;
    cellH += 1;
  }
  bufferRed = (float *)malloc(sizeof(float) * cellW * cellH);
  bufferGreen = (float *)malloc(sizeof(float) * cellW * cellH);
  bufferBlue = (float *)malloc(sizeof(float) * cellW * cellH);
  counter = (int *)malloc(sizeof(float)* cellW * cellH);
  // initialize burffer
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      bufferRed[buffer_i*cellW + buffer_j] = 0;
      bufferGreen[buffer_i*cellW + buffer_j] = 0;
      bufferBlue[buffer_i*cellW + buffer_j] = 0;
      counter[buffer_i*cellW + buffer_j] = 0;
    }
  }

  // sum the value on cell
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      for (i = 0; i < (int)cSize; i++) {
        for (j = 0; j < (int)cSize; j++) {
          if (particle_cellH != 0 && ((int)cSize * buffer_i + i) >= (int)(*h))
            continue;
          else if (particle_cellW != 0 && ((int)cSize * buffer_j + j) >= (int)(*w))
            continue;
          else{
            rgbIndex = (buffer_i * (int)cSize + i) * (int)(*w) + buffer_j * (int)cSize +j;
            bufferRed[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].red);
            bufferGreen[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].green);
            bufferBlue[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].blue);
            counter[buffer_i*cellW + buffer_j] += 1;
          }
        }
      }
    }
  }

  // average
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      bufferRed[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
      bufferGreen[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
      bufferBlue[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
    }
  }

  // resize as original size
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      for (i = 0; i < (int)cSize; i++) {
        for (j = 0; j < (int)cSize; j++) {
          if (particle_cellH != 0 && ((int)cSize * buffer_i + i) >= (int)(*h))
            continue;
          else if (particle_cellW != 0 && ((int)cSize * buffer_j + j) >= (int)(*w))
            continue;
          else{
            rgbIndex = (buffer_i * (int)cSize + i) * (int)(*w) + buffer_j * (int)cSize +j;
            rgb[rgbIndex].red = (unsigned char)bufferRed[buffer_i*cellW + buffer_j];
            rgb[rgbIndex].green = (unsigned char)bufferGreen[buffer_i*cellW + buffer_j];
            rgb[rgbIndex].blue = (unsigned char)bufferBlue[buffer_i*cellW + buffer_j];
          }
        }
      }
    }
  }

  return SUCCESS;
}

// mosaic calculation for multiple thread, implementation of openmp
int mosaicCal_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize)
{
  int i,j,buffer_i, buffer_j, rgbIndex;
  int cellW, cellH, particle_cellH, particle_cellW;
  int *counter;
  float *bufferRed, *bufferGreen, *bufferBlue;
  //* Calculation of masic filter
  cellW = (*w) / (int)cSize;
  cellH = (*h) / (int)cSize;
  particle_cellW = (*w) % (int)cSize;
  particle_cellH = (*h) % (int)cSize;
  if (cSize < 0 || cSize >= (*w) || cSize >= (*h)) {
		fprintf(stderr, "Calculation Error: masic filter size is too large\n");
    return FAILURE;
  }
  // consider having particle mosaic cell
  if (particle_cellW != 0 || particle_cellH != 0) {
    cellW += 1;
    cellH += 1;
  }
  bufferRed = (float *)malloc(sizeof(float) * cellW * cellH);
  bufferGreen = (float *)malloc(sizeof(float) * cellW * cellH);
  bufferBlue = (float *)malloc(sizeof(float) * cellW * cellH);
  counter = (int *)malloc(sizeof(float)* cellW * cellH);
  // initialize burffer
  #pragma omp parallel for collapse(2)
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      bufferRed[buffer_i*cellW + buffer_j] = 0;
      bufferGreen[buffer_i*cellW + buffer_j] = 0;
      bufferBlue[buffer_i*cellW + buffer_j] = 0;
      counter[buffer_i*cellW + buffer_j] = 0;
    }
  }

  // sum the value on cell
  #pragma omp parallel for collapse(4)
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      for (i = 0; i < (int)cSize; i++) {
        for (j = 0; j < (int)cSize; j++) {
          if (particle_cellH != 0 && ((int)cSize * buffer_i + i) >= (int)(*h))
            continue;
          else if (particle_cellW != 0 && ((int)cSize * buffer_j + j) >= (int)(*w))
            continue;
          else{
            rgbIndex = (buffer_i * (int)cSize + i) * (int)(*w) + buffer_j * (int)cSize +j;
            bufferRed[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].red);
            bufferGreen[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].green);
            bufferBlue[buffer_i*cellW + buffer_j] += (float)(rgb[rgbIndex].blue);
            counter[buffer_i*cellW + buffer_j]++;
          }
        }
      }
    }
  }

  // average
  #pragma omp parallel for collapse(2)
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      bufferRed[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
      bufferGreen[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
      bufferBlue[buffer_i*cellW + buffer_j] /= (float)counter[buffer_i*cellW+buffer_j];
    }
  }

  // resize as original size
  #pragma omp parallel for collapse(4)
  for (buffer_i = 0; buffer_i < cellH; buffer_i++) {
    for (buffer_j = 0; buffer_j < cellW; buffer_j++) {
      for (i = 0; i < (int)cSize; i++) {
        for (j = 0; j < (int)cSize; j++) {
          if (particle_cellH != 0 && ((int)cSize * buffer_i + i) >= (int)(*h))
            continue;
          else if (particle_cellW != 0 && ((int)cSize * buffer_j + j) >= (int)(*w))
            continue;
          else{
            rgbIndex = (buffer_i * (int)cSize + i) * (int)(*w) + buffer_j * (int)cSize +j;
            rgb[rgbIndex].red = (unsigned char)bufferRed[buffer_i*cellW + buffer_j];
            rgb[rgbIndex].green = (unsigned char)bufferGreen[buffer_i*cellW + buffer_j];
            rgb[rgbIndex].blue = (unsigned char)bufferBlue[buffer_i*cellW + buffer_j];
          }
        }
      }
    }
  }
  return SUCCESS;
}

// calculate average rgb value in single thread (CPU)
struct PPMrgb average_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d)
{
  struct PPMrgb buffer;
  float bufferRed = 0, bufferGreen = 0, bufferBlue = 0;
  int i;

  // calculate average
  for (i = 0; i < (*w) * (*h); i++) {
    bufferRed += (float)rgb[i].red;
    bufferGreen += (float)rgb[i].green;
    bufferBlue += (float)rgb[i].blue;
  }
  buffer.red = (unsigned char) (bufferRed / ((*w)*(*h)));
  buffer.green = (unsigned char) (bufferGreen / ((*w)*(*h)));
  buffer.blue = (unsigned char) (bufferBlue / ((*w)*(*h)));

  return buffer;
}

// calculate average rgb value in multiple threads (openMP)
struct PPMrgb average_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d)
{
  struct PPMrgb buffer;
  float bufferRed = 0, bufferGreen = 0, bufferBlue = 0;
  int i;

  // calculate average
  #pragma omp parallel for private(i) reduction(+:bufferRed) reduction(+:bufferGreen) reduction(+:bufferBlue)
  for (i = 0; i < (*w) * (*h); i++) {
    bufferRed += (float)rgb[i].red;
    bufferGreen += (float)rgb[i].green;
    bufferBlue += (float)rgb[i].blue;
  }

  buffer.red = (unsigned char) (bufferRed / ((*w)*(*h)));
  buffer.green = (unsigned char) (bufferGreen / ((*w)*(*h)));
  buffer.blue = (unsigned char) (bufferBlue / ((*w)*(*h)));

  return buffer;
}

int cudaMosaic_cal()
{
  struct PPMrgb *ppmrgb_input;
  struct PPMrgb *cudargb;
  unsigned int *width_, *height_, *deepth_;
  unsigned int magic_number;
  float *r_float, *g_float, *b_float;
  float *r_output, *g_output, *b_output;
  float *r_sum, *g_sum, *b_sum;
  float width_float, height_float;
  float c_float = (float)c;
  float msec;
  cudaEvent_t cuda_start, cuda_stop;

  printf("Start mosaic calculation in cuda\n");
  // host memory
  width_  = (unsigned int*)malloc(sizeof(unsigned int));
  height_ = (unsigned int*)malloc(sizeof(unsigned int));
  deepth_ = (unsigned int*)malloc(sizeof(unsigned int));
  r_sum = (float *)malloc(sizeof(float));
  g_sum = (float *)malloc(sizeof(float));
  b_sum = (float *)malloc(sizeof(float));

  // host memory 2, after read file
  ppmrgb_input = readPPMFile(input_fileName, width_, height_, deepth_);
  cudargb = (struct PPMrgb*)malloc(sizeof(struct PPMrgb)* (*width_) * (*height_));
  r_float = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  g_float = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  b_float = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  r_output = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  g_output = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  b_output = (float *)malloc(sizeof(float)*(*width_)*(*height_));
  for(int i=0; i<(*width_)*(*height_); i++){
    r_float[i] = (float)ppmrgb_input[i].red;
    g_float[i] = (float)ppmrgb_input[i].green;
    b_float[i] = (float)ppmrgb_input[i].blue;
  }

  //printf("%d %d %d\n", *width_, *height_, *deepth_);
  width_float = (float)(*width_);
  height_float = (float)(*height_);

  // prepare for device memory
  float *d_width_float, *d_height_float, *d_c_float;
  float *d_r_float, *d_g_float, *d_b_float;
  float *d_r_output, *d_g_output, *d_b_output;
  float *d_r_sum, *d_g_sum, *d_b_sum;
  // set pointer to float type variables
  float *p_c_float = &c_float;
  float *p_width_float = &width_float;
  float *p_height_float = &height_float;

  // allocate device memory
  cudaMalloc((void **)&d_c_float, sizeof(float));
  cudaMalloc((void **)&d_width_float, sizeof(float));
  cudaMalloc((void **)&d_height_float, sizeof(float));
  cudaMalloc((void **)&d_r_float, sizeof(float)*(*width_)*(*height_));
  cudaMalloc((void **)&d_g_float, sizeof(float)*(*width_)*(*height_));
  cudaMalloc((void **)&d_b_float, sizeof(float)*(*width_)*(*height_));
  checkCUDAError("CUDA malloc for must use");

  // copy necessary value to device
  cudaMemcpy(d_c_float, p_c_float, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_width_float, p_width_float, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_height_float, p_height_float, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_r_float, r_float, sizeof(float)*(*width_)*(*height_), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g_float, g_float, sizeof(float)*(*width_)*(*height_), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b_float, b_float, sizeof(float)*(*width_)*(*height_), cudaMemcpyHostToDevice);
  checkCUDAError("CUDA memcpy for must use");

  // set block size and grid size of average value
  int threads_num = 32;
  int blocks_num = (*width_)*(*height_) / threads_num;
  // set block size and grid size
  dim3 threads_pre_block, blocks_pre_grid;
  if ( 0 < c && c <= 8) {
    threads_pre_block.x = 8;
    threads_pre_block.y = 8;
  }else if (8 < c && c <= 16) {
    threads_pre_block.x = 16;
    threads_pre_block.y = 16;
  }else if (16 < c && c <= 32) {
    threads_pre_block.x = THREADS_PRE_BLOCK_X_32;
    threads_pre_block.y = THREADS_PRE_BLOCK_Y_32;
  }else{
		fprintf(stderr, "Error: CUDA Implementation not support for over 32X32 mosaic size.\n");
		print_help();
    return FAILURE;
  }
  blocks_pre_grid.x = ((*width_) + c - 1) / c;
  blocks_pre_grid.y = ((*height_) + c - 1) / c;
  // allocate shared memory size
  int memsize = sizeof(float) * (threads_pre_block.x * threads_pre_block.y);

  // calculate average rgb value
  cudaMalloc((void **)&d_r_sum, sizeof(float));
  cudaMalloc((void **)&d_g_sum, sizeof(float));
  cudaMalloc((void **)&d_b_sum, sizeof(float));
  checkCUDAError("CUDA malloc for output sum");

  // allocate device memory for mosaic calculate
  cudaMalloc((void **)&d_r_output, sizeof(float)*(*width_)*(*height_));
  cudaMalloc((void **)&d_g_output, sizeof(float)*(*width_)*(*height_));
  cudaMalloc((void **)&d_b_output, sizeof(float)*(*width_)*(*height_));
  checkCUDAError("CUDA malloc for mosaic output");

  // Allocate cuda events for timing
  cudaEventCreate(&cuda_start);
  cudaEventCreate(&cuda_stop);
  checkCUDAError("CUDA event creation");

  // count time and mosaic calculation in RGB value
  cudaEventRecord(cuda_start);
  // start calculate seperated by RGB value
  average_cuda<<<blocks_num, threads_num, sizeof(float)*32>>>(d_r_float, d_r_sum, d_width_float, d_height_float);
  checkCUDAError("average for red");
  average_cuda<<<blocks_num, threads_num, sizeof(float)*32>>>(d_g_float, d_g_sum, d_width_float, d_height_float);
  checkCUDAError("average for green");
  average_cuda<<<blocks_num, threads_num, sizeof(float)*32>>>(d_b_float, d_b_sum, d_width_float, d_height_float);
  checkCUDAError("average for blue");
  // copy value from device to host
  cudaMemcpy(r_sum, d_r_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(g_sum, d_g_sum, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(b_sum, d_b_sum, sizeof(float), cudaMemcpyDeviceToHost);
  checkCUDAError("CUDA memcpy for output");
  printf("The cuda red = %.1f, green = %.1f, blue = %.1f\n", (*r_sum)/(width_float*height_float), (*g_sum)/(width_float*height_float), (*b_sum)/(width_float*height_float));
  // free average calculation memory
  cudaFree(d_r_sum);cudaFree(d_g_sum);cudaFree(d_b_sum);
  free(r_sum);free(g_sum);free(b_sum);

  // start to mosaic calculation
  mosaicCal_cuda<<<blocks_pre_grid, threads_pre_block, memsize>>>(d_r_float, d_r_output, d_width_float, d_height_float, d_c_float);
  checkCUDAError("mosaic red");
  mosaicCal_cuda<<<blocks_pre_grid, threads_pre_block, memsize>>>(d_g_float, d_g_output, d_width_float, d_height_float, d_c_float);
  checkCUDAError("mosaic green");
  mosaicCal_cuda<<<blocks_pre_grid, threads_pre_block, memsize>>>(d_b_float, d_b_output, d_width_float, d_height_float, d_c_float);
  checkCUDAError("mosaic blue");
  cudaEventRecord(cuda_stop);
  cudaEventSynchronize(cuda_stop);
  checkCUDAError("CUDA kernel execution and timing");
  cudaEventElapsedTime(&msec, cuda_start, cuda_stop);
  cudaThreadSynchronize();
  checkCUDAError("CUDA timing");

  // copy device memory to host
  cudaMemcpy(r_output, d_r_output, sizeof(float)*(*width_)*(*height_), cudaMemcpyDeviceToHost);
  cudaMemcpy(g_output, d_g_output, sizeof(float)*(*width_)*(*height_), cudaMemcpyDeviceToHost);
  cudaMemcpy(b_output, d_b_output, sizeof(float)*(*width_)*(*height_), cudaMemcpyDeviceToHost);
  checkCUDAError("CUDA memcpy for output to host");

  // write back to PPM size
  for(int i=0; i<(*width_)*(*height_); i++){
    cudargb[i].red = (unsigned char)r_output[i];
    cudargb[i].green = (unsigned char)g_output[i];
    cudargb[i].blue = (unsigned char)b_output[i];
  }
  printf("The cuda time is: %.2fms\n", msec);
  if (writePPMFile(output_fileName, cudargb, width_, height_, deepth_, ppm_format) == FAILURE)
    return FAILURE;

  // free all
  cudaFree(d_width_float);cudaFree(d_height_float);
  cudaFree(d_c_float);cudaFree(d_r_float);cudaFree(d_g_float);cudaFree(d_b_float);
  cudaFree(d_r_output);cudaFree(d_g_output);cudaFree(d_b_output);
  freeAll(ppmrgb_input, width_, height_, deepth_);
  printf("done\n");
  return SUCCESS;
}

// check error
void checkCUDAError(const char  *msg)
{
  cudaError_t err  = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// free variables memory
void freeAll(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d)
{
  free(rgb); free(w); free(h); free(d);
}
