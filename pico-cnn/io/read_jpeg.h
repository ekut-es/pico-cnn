/** 
 * @brief reads a jpeg file and stores it into a 2-dimensional (1-dimensional
 * arrays for R,G,B) array
 * Adopted from Denis Tola:
 * https://www.quora.com/In-C-and-C%2B%2B-how-can-I-open-an-image-file-like-JPEG-and-read-it-as-a-matrix-of-pixels/answer/Denis-Tola?srid=uaBxY
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_JPEG_H
#define READ_JPEG_H

#include "../parameters.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <jpeglib.h>

/**
 * @brief reads a jpeg file and stores it into a 2-dimensional (1-dimensional
 * arrays for R,G,B) array
 *
 * @param image 2D-array which contains the image data (will be allocated inside) 
 * [0] = red, [1] = green, [2] = blue 
 * @param jpeg_path full path to jpeg image which should be read
 * @param padding which should be added to the edges (lower_bound value)
 * @param lower_bound of range to which a pixel should be scaled
 * @param upper_bound of range to which a pixel should be scaled
 *
 * @return error (0 = success, 1 = error)
 */
int read_jpeg(fp_t*** image, const char* jpeg_path, const fp_t lower_bound, const fp_t upper_bound, uint16_t *height, uint16_t *width) {

  	FILE * jpeg_file;

	jpeg_file = fopen(jpeg_path, "rb");

    if(jpeg_file != 0) {
		uint8_t r,g,b;
        fp_t range = fabs(lower_bound - upper_bound);

		struct jpeg_decompress_struct cinfo;
		struct jpeg_error_mgr jerr;

  		JSAMPARRAY jpeg_buffer;
		uint16_t row_stride;
		unsigned int pos = 0;

		cinfo.err = jpeg_std_error(&jerr);
  		jpeg_create_decompress(&cinfo);
  		jpeg_stdio_src(&cinfo, jpeg_file);
  		(void) jpeg_read_header(&cinfo, TRUE);
  		(void) jpeg_start_decompress(&cinfo);

  		*width = cinfo.output_width;
  		*height = cinfo.output_height;

		(*image) = (fp_t**) malloc(3*sizeof(fp_t*));
		// red
		(*image)[2] = (fp_t*) malloc((*height)*(*width)*sizeof(fp_t));
		// green
		(*image)[1] = (fp_t*) malloc((*height)*(*width)*sizeof(fp_t));
		// blue
		(*image)[0] = (fp_t*) malloc((*height)*(*width)*sizeof(fp_t));

 		row_stride = (*width) * cinfo.output_components;
  		jpeg_buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	  	while (cinfo.output_scanline < cinfo.output_height) {

			(void) jpeg_read_scanlines(&cinfo, jpeg_buffer, 1);

            int x;

			for(x = 0; x < *width; x++) {
				r = jpeg_buffer[0][cinfo.output_components * x];
			  	if (cinfo.output_components > 2) {
					g = jpeg_buffer[0][cinfo.output_components * x + 1];
					b = jpeg_buffer[0][cinfo.output_components * x + 2];
			  	} else {
					g = r;
					b = r;
			  	}
				(*image)[2][pos] = (((r / 255.0f) * range) + lower_bound);
				(*image)[1][pos] = (((g / 255.0f) * range) + lower_bound);
				(*image)[0][pos] = (((b / 255.0f) * range) + lower_bound);
				pos++;
			}
		}
		fclose(jpeg_file);
		(void) jpeg_finish_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);

		return 0;
	}

    return 1;
}

#endif // READ_JPEG_H
