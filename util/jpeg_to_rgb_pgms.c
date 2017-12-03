/** 
 * @brief simple program which reads a JPEG and stores R,G,B as PGMs
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define DEBUG

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if(argc != 2) {
        fprintf(stderr, "no path to jpeg file provided!\n");
        return 1;
    }

    fp_t** jpeg_image;

    uint16_t height;
    uint16_t width;


	char filename[strlen(argv[1])];

	char filename_red[strlen(argv[1])+16];
	char filename_blue[strlen(argv[1])+16];
	char filename_green[strlen(argv[1])+16];

	char* c;
	int pos;

	c = strrchr(argv[1], '.');
	pos = c-argv[1];
	strncpy(filename, argv[1], pos);
	filename[pos] = '\0';
	
	strcpy(filename_red, filename);
	strcat(filename_red, "_red.pgm");

	strcpy(filename_blue, filename);
	strcat(filename_blue, "_blue.pgm");

	strcpy(filename_green, filename);
	strcat(filename_green, "_green.pgm");

    read_jpeg(&jpeg_image, argv[1], 0.0, 1.0, &height, &width);

	write_pgm(jpeg_image[2], height, width, filename_red);
	printf("%s\n", filename_red);

	write_pgm(jpeg_image[1], height, width, filename_blue);
	printf("%s\n", filename_blue);

	write_pgm(jpeg_image[0], height, width, filename_green);
	printf("%s\n", filename_green);

	free(jpeg_image[0]);
	free(jpeg_image[1]);
	free(jpeg_image[2]);

    return 0;
} 
