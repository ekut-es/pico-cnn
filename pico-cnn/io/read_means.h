/** 
 * @brief provides functions to read means previously generated from
 * tiny-dnn/caffe CNNs
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_MEANS_H
#define READ_MEANS_H

#include "../parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/**
 * @brief reads three means from a means file
 * Find the definition of the file format in doc/mean_file_format.md
 * 
 * @param path_to_means_file file in which the means are stored
 * @param 1D-array of fp_t (size 3) to store the means
 * 
 * @return =! 0 means an error occured
 */
int read_means(const char* path_to_means_file, fp_t* means) {
    FILE *means_file;
    means_file = fopen(path_to_means_file, "r");


    if(means_file != 0) {

        char buffer[100];

        // read magic number
        fgets(buffer, 100, means_file);

        if(strcmp(buffer,"FD\n") != 0) {
            fclose(means_file);
            return 1;
        }

        float mean;
        fscanf(means_file, "%a\n", &mean);
        means[0] = mean;

        fscanf(means_file, "%a\n", &mean);
        means[1] = mean;

        fscanf(means_file, "%a\n", &mean);
        means[2] = mean;

        fclose(means_file);

        return 0;
    }

    return 1;
}



#endif // READ_MEANS_H
