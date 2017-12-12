/** 
 * @brief benchmark for different fully connected implementations
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define DEBUG

#define OUTPUT_SIZE 512

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>
#include <omp.h>

enum mode_t {NAIVE, CPU, CGRA, GPU};

void usage() {
    printf("./fully_connected_benchmark MODE PATH_TO_INPUT_IMAGE\n");
    printf("\tMODE: naive, cpu, cgra, gpu\n");
    printf("\tPATH_TO_INPUT_IMAGE: *.pgm, *.{jpg,jpeg,JPG,JPEG}\n");
}

int main(int argc, char** argv) {
    if(argc != 3) {
        fprintf(stderr, "no mode/path to input image provided!\n");
        usage();
        return 1;
    }

    // check if naive of opt was chosen
    mode_t mode = NAIVE;

    if(strcmp(argv[1], "naive") == 0) {
        mode = NAIVE;
    } else if(strcmp(argv[1], "cpu") == 0) {
        mode = CPU;
    } else if(strcmp(argv[1], "cgra") == 0) {
        mode = CGRA;
    } else if(strcmp(argv[1], "gpu") == 0) {
        mode = GPU;
    }

    // check if input image is pgm or jpeg
    uint8_t jpg = 0;
    char * extension;

    extension = strrchr(argv[2], '.');
    
    if(strcmp(extension+1, "pgm") == 0) {
        jpg = 0;
    } else if(strcmp(extension+1, "jpg") == 0 || strcmp(extension+1, "jpeg") == 0 || strcmp(extension+1, "JPG") == 0 || strcmp(extension+1, "JPEG") == 0) {
        jpg = 1;
    }

    uint16_t height, width;
    fp_t* input_image;

    if(jpg == 0) {
        fp_t* input_image_pgm;
        if(read_pgm(&input_image_pgm, argv[2], 0, 0.0, 1.0, &height, &width) != 0) {
            fprintf(stderr, "could not read pgm image '%s'!\n", argv[1]);
            return 1;
        }
        input_image = input_image_pgm;
    } else if(jpg == 1) {
        fp_t** input_image_jpg;
        if(read_jpeg(&input_image_jpg, argv[2], 0.0, 1.0, &height, &width) != 0) {
            fprintf(stderr, "could not read jpeg image '%s'!\n", argv[1]);
            return 1;
        }
        input_image = input_image_jpg[0];
        free(input_image_jpg[1]);
        free(input_image_jpg[2]);
    }

    #ifdef DEBUG
    write_pgm(input_image, height, width, "input.pgm");
    #endif

    int i; 
    fp_t value;

    // generate kernel
    fp_t* kernel = (fp_t*) malloc(height*width*OUTPUT_SIZE*sizeof(fp_t));

    value = -1.0;

    for(i = 0; i < height*width*OUTPUT_SIZE; i++) {
        value += 0.1;

        kernel[i] = value;

        if(value > 0.9) {
            value = -1.0;
        }
    }

    // generate bias
    fp_t* bias = (fp_t*) malloc(OUTPUT_SIZE*sizeof(fp_t));

    value = -1.0;
    for(i = 0; i < OUTPUT_SIZE; i++) {
        value += 0.1;

        bias[i] = value;

        if(value > 0.9) {
            value = -1.0;
        }
    }

    fp_t* output_image = (fp_t*) malloc(OUTPUT_SIZE*sizeof(fp_t));

    if(mode == NAIVE) {
        fully_connected_naive(input_image, height*width, output_image, OUTPUT_SIZE, kernel, bias);
    } else if(mode == CPU) {
        restructure_fully_connected_kernel(&kernel, height*width, OUTPUT_SIZE);
        #pragma omp parallel num_threads(4) 
        {
            #pragma omp sections 
            {
                #pragma omp section
                fully_connected_cpu(input_image, height*width, output_image, OUTPUT_SIZE, kernel, bias, 0, 1*(OUTPUT_SIZE/4));
                #pragma omp section
                fully_connected_cpu(input_image, height*width, output_image, OUTPUT_SIZE, kernel, bias, 1*(OUTPUT_SIZE/4), 2*(OUTPUT_SIZE/4));
                #pragma omp section
                fully_connected_cpu(input_image, height*width, output_image, OUTPUT_SIZE, kernel, bias, 2*(OUTPUT_SIZE/4), 3*(OUTPUT_SIZE/4));
                #pragma omp section
                fully_connected_cpu(input_image, height*width, output_image, OUTPUT_SIZE, kernel, bias, 3*(OUTPUT_SIZE/4), OUTPUT_SIZE);
            }
        }
    }

    #ifdef DEBUG
    if(mode == NAIVE) {
        write_pgm(output_image, 1, OUTPUT_SIZE, "naive.pgm");
        write_float(output_image, 1, OUTPUT_SIZE, "naive.float");
    } else if(mode == CPU) {
        write_pgm(output_image, 1, OUTPUT_SIZE, "cpu.pgm");
        write_float(output_image, 1, OUTPUT_SIZE, "cpu.float");
    }
    #endif


    free(kernel);
    free(bias);
    free(input_image);
    free(output_image);
}
