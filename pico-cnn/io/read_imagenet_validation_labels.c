#include "read_imagenet_validation_labels.h"

int read_imagenet_validation_labels(const char* path_to_imagenet_validation_labels, uint32_t** imagenet_validation_labels, uint32_t num_labels) {
    FILE *labels_file;
    labels_file = fopen(path_to_imagenet_validation_labels, "r");

	if(labels_file != 0) {

        (*imagenet_validation_labels) = (uint32_t*) malloc((num_labels+1)*sizeof(uint32_t));

        char buffer[255];
        char label_buffer[6];
        int i = 1;

        while(fgets(buffer, 255, labels_file) && i < num_labels) {
            memcpy(label_buffer, &buffer[FILE_NAME_LENGTH+1], 4);
            (*imagenet_validation_labels)[i] = (uint32_t) atoi(label_buffer);
            i++;
        }

        fclose(labels_file);
        return i;
    }

    return 0;
}
