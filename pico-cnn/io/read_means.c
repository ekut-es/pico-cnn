#include "read_means.h"

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
