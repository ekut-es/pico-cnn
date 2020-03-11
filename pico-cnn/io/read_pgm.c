#include "read_pgm.h"

int32_t read_pgm(fp_t** image, const char* pgm_path, const uint8_t padding, const fp_t lower_bound, const fp_t upper_bound, uint16_t* height, uint16_t* width) {

    FILE *pgm_file;
    pgm_file = fopen(pgm_path, "r");

    if(pgm_file != 0) {

        char buffer[100];

        fgets(buffer, 100, pgm_file);

        if(strcmp(buffer,"P5\n") != 0) {
            fclose(pgm_file);
            return 1;
        }

        uint8_t max_gray_value;

        fp_t range = fabs(lower_bound - upper_bound);

        // skip comment
        fgets(buffer, 100, pgm_file);
        // read dimensions
        fscanf(pgm_file, "%hu %hu\n", width, height);
        // read max gray value
        fscanf(pgm_file, "%hhu\n", &max_gray_value);

        (*image) = (fp_t*) malloc(((*height)+2*padding) * ((*width)+2*padding) * sizeof(fp_t));

        uint16_t row, column;
        uint8_t pixel;

        for(row = 0; row < (*height)+2*padding; row++) {
            for(column = 0; column < (*width)+2*padding; column++) {
                if(row < padding || row >= (*height)+padding) {
                    (*image)[row*((*width)+2*padding)+column] = lower_bound;
                } else if(column < padding || column >= (*width)+padding) {
                    (*image)[row*((*width)+2*padding)+column] = lower_bound;
                } else {
                    pixel = (uint8_t) fgetc(pgm_file);
                    (*image)[row*((*width)+2*padding)+column] = (((pixel / 255.0f) * range) + lower_bound);
                }
            }
        }

        fclose(pgm_file);

        return 0;
    }

    return 1;
}
