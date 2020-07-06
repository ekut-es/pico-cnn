#include "read_imagenet_labels.h"

int32_t read_imagenet_labels(const char* path_to_imagenet_labels, char*** imagenet_labels, uint32_t num_labels) {
    FILE *labels_file;
    labels_file = fopen(path_to_imagenet_labels, "r");

    if(labels_file != nullptr) {

        (*imagenet_labels) = (char**) malloc(1000*sizeof(char*));

        char buffer[255];
        int32_t i = 0;
        int32_t label_length;

        while(fgets(buffer, 255, labels_file)) {
            label_length = strlen(buffer);
            (*imagenet_labels)[i] = (char*) malloc((label_length)*sizeof(char));
            strncpy((*imagenet_labels)[i], buffer, label_length-1);
            (*imagenet_labels)[i][label_length-1] = '\0';
            i++;
        }

        fclose(labels_file);
        return i;
    }

    return 0;
}
