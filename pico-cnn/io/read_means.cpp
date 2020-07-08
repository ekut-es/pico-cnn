#include "read_means.h"

int32_t read_means(const char* path_to_means_file, fp_t* means) {
    FILE *means_file;
    means_file = fopen(path_to_means_file, "r");


    if(means_file != 0) {

        char buffer[100];

        // read magic number
        if(fgets(buffer, 100, means_file) == nullptr){
            PRINT_ERROR("Failed to read magic number.")
            fclose(means_file);
            return 1;
        }

        if(strcmp(buffer,"FD\n") != 0) {
            PRINT_ERROR("Wrong magic number.")
            fclose(means_file);
            return 1;
        }

        //fread(means, sizeof(fp_t), 3, means_file);

        fp_t mean;
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
