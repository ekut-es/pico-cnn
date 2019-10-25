#include "read_jpeg.h"

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
				/*
				(*image)[2][pos] = (((r / 255.0f) * range) + lower_bound);
				(*image)[1][pos] = (((g / 255.0f) * range) + lower_bound);
				(*image)[0][pos] = (((b / 255.0f) * range) + lower_bound);
				*/
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
