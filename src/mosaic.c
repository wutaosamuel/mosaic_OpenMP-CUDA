/*
* This is the assignment 1 program code
* Student name is Tao Wu
* Email is twu17@sheffield.ac.uk
* Student Register No. is 180127601
* I declare that this code is my own work
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <omp.h>

#define FAILURE 0
#define SUCCESS !FAILURE

#define USER_NAME "acp18tw"		//replace with your user name

typedef enum MODE { CPU, OPENMP, CUDA, ALL } MODE;
typedef enum PPM_FORMAT { PPM_BINARY, PPM_PLAIN_TEXT } PPM_FORMAT;

struct PPMrgb{
  unsigned char red;
  unsigned char green;
  unsigned char blue;
};

void print_help();
int process_command_line(int argc, char *argv[]);

// self function declaretion
unsigned int checkPowerOfTwo(unsigned int x);
struct PPMrgb* readPPMFile(char *filename, unsigned int *w, unsigned int *h, unsigned int *d);
struct PPMrgb average_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);
struct PPMrgb average_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);
int writePPMFile(char *filename, struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, PPM_FORMAT format);
int mosaicCal_single(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize);
int mosaicCal_multiple(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d, unsigned int cSize);
void freeAll(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d);

// globe variables declaretion
unsigned int c = 0;
char *input_fileName, *output_fileName;
MODE execution_mode = CPU;
PPM_FORMAT ppm_format = PPM_BINARY;

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
			break;
		}
		case (CUDA) : {
		  fprintf(stderr, "Error: CUDA Implementation not required for assignment part 1.\n");
		  print_help();
      freeAll(ppmrgb, width, height, deepth);
	    free(input_fileName);
	    free(output_fileName);
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
			break;
		}
	}

	//save the output image file (from last executed mode)
  if (writePPMFile(output_fileName, ppmrgb, width, height, deepth, ppm_format) == FAILURE)
    return 1;

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
	if (strncasecmp(argv[2], "CPU", 3) == 0)
		execution_mode = CPU;
	else if (strncasecmp(argv[2], "OPENMP", 6) == 0)
		execution_mode = OPENMP;
	else if (strncasecmp(argv[2], "CUDA", 4) == 0)
		execution_mode = CUDA;
	else if (strncasecmp(argv[2], "ALL", 3) == 0)
		execution_mode = ALL;
	else {
		fprintf(stderr, "Error: check entered mode. Correct usage is...\n");
		print_help();
		return FAILURE;
	}

	//TODO: read in the input image name
	if (strcmp(argv[3], "-i") != 0){
		fprintf(stderr, "Error: unknow argument. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	int input_fileCount = strlen(argv[4]);
	input_fileName = (char *) malloc(input_fileCount * sizeof(char));
	strcpy(input_fileName, argv[4]);
	// check if file exists
  if (access(input_fileName, 0) == -1){
		fprintf(stderr, "Error: intput file not exist");
		return FAILURE;
	}

	//TODO: read in the output image name
	if (strcmp(argv[5], "-o") != 0){
		fprintf(stderr, "Error: unknow argument. Correct usage is...\n");
		print_help();
		return FAILURE;
	}
	int output_fileCount = strlen(argv[6]);
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
		  if (strncasecmp(argv[8], "PPM_BINARY", 10) == 0)
			   ppm_format = PPM_BINARY;
		  else if (strncasecmp(argv[8], "PPM_PLAIN_TEXT", 12) == 0)
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

// free variables memory
void freeAll(struct PPMrgb *rgb, unsigned int *w, unsigned int *h, unsigned int *d)
{
  free(rgb); free(w); free(h); free(d);
}
