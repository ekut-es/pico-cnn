/** 
 * @brief provides function to read the ImageNet labels
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#ifndef READ_IMAGENET_LABELS_H
#define READ_IMAGENET_LABELS_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief reads ImageNet labels from a given file
 *
 * @param path_to_imagenet_label 
 * @param imagenet_labels will be allocated inside
 * @param num_labels number of labels which should be read from file
 *
 * @return 0 = error occured
 */
int read_imagenet_labels(const char* path_to_imagenet_labels, char*** imagenet_labels, uint32_t num_labels) {
    FILE *labels_file;
    labels_file = fopen(path_to_imagenet_labels, "r");

	if(labels_file != 0) {

        (*imagenet_labels) = (char**) malloc(1000*sizeof(char*));

        char* buffer = NULL;
        int i = 0;
        int label_length;
        size_t len = 0;

        while (((label_length = getline(&buffer, &len, labels_file)) != -1) && i < num_labels) {
            (*imagenet_labels)[i] = (char*) malloc((label_length-1)*sizeof(char));
            strncpy((*imagenet_labels)[i], buffer, label_length-1);
            (*imagenet_labels)[i][label_length-1] = '\0';
            i++;
        }

        fclose(labels_file);
        return i;
    }

    return 0;
}

#endif // READ_IMAGENET_LABELS_H

