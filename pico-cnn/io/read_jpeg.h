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
int32_t read_jpeg(fp_t*** image, const char* jpeg_path, const fp_t lower_bound, const fp_t upper_bound, uint16_t *height, uint16_t *width);

#endif // READ_JPEG_H
