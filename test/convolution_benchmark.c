/** 
 * @brief benchmark for different implementations of 2D convolution
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define JPEG
#define DEBUG

#include "pico-cnn/pico-cnn.h"
#include <stdio.h>


#define KERNEL_SIZE 11
#define KERNEL_CROP (KERNEL_SIZE/2)

/*
// kernel 3x3
fp_t kernel[KERNEL_SIZE*KERNEL_SIZE] = {
    0.1933171948, 0.0535442412, 0.1878331911, 
    0.0004219844, 0.2196035127, 0.0857518851, 
    0.0956728014, 0.1089575281, 0.0548976613
};
*/

/*
// kernel 5x5
fp_t kernel[KERNEL_SIZE*KERNEL_SIZE] = {
    0.04084506, 0.04182326, 0.03507044, 0.04544337, 0.03872346, 
    0.03757646, 0.04150467, 0.04431239, 0.03268669, 0.04032026,
    0.04189709, 0.04434769, 0.03836922, 0.03769494, 0.04838630, 
    0.03593228, 0.03419258, 0.03564377, 0.04019063, 0.04144708, 
    0.03534928, 0.04342522, 0.03947366, 0.04237048, 0.04297373
};
*/

// kernel 11x11
fp_t kernel[KERNEL_SIZE*KERNEL_SIZE] = {
    0.0133775946, 0.0081662832, 0.0037958851, 0.0025589938, 0.0111077079, 0.0094678558, 0.0084671443, 0.0048633728, 0.0105224367, 0.0022854223, 0.0067605625, 
    0.0122735694, 0.0077536908, 0.0092950599, 0.0034780756, 0.0042276106, 0.0047869384, 0.0122461177, 0.0020111278, 0.0136192431, 0.0074915024, 0.0014510841, 
    0.0048870096, 0.0107253250, 0.0028913983, 0.0060770306, 0.0077741476, 0.0108696328, 0.0137626939, 0.0082248597, 0.0096482358, 0.0095305894, 0.0032545504, 
    0.0159774844, 0.0005557182, 0.0104320559, 0.0119777216, 0.0085896735, 0.0132704344, 0.0095271051, 0.0033663883, 0.0083837807, 0.0043944058, 0.0062804474, 
    0.0078444700, 0.0076076714, 0.0155442932, 0.0099033487, 0.0112353396, 0.0006153568, 0.0047091491, 0.0124721280, 0.0092239241, 0.0072302259, 0.0128413587, 
    0.0056841725, 0.0033638936, 0.0153741250, 0.0118403007, 0.0060344246, 0.0117556617, 0.0110957269, 0.0046839545, 0.0116901183, 0.0143944550, 0.0085295070, 
    0.0044643005, 0.0105836846, 0.0036357740, 0.0041731376, 0.0091646626, 0.0122367628, 0.0049231543, 0.0148439477, 0.0051299223, 0.0097820201, 0.0038243508,
    0.0072066147, 0.0094741429, 0.0158191804, 0.0052030003, 0.0015101179, 0.0110555604, 0.0106454520, 0.0021575936, 0.0161685845, 0.0046176961, 0.0031489035, 
    0.0083015171, 0.0029790727, 0.0018164087, 0.0131506898, 0.0053539603, 0.0098669754, 0.0158725144, 0.0157118653, 0.0005730547, 0.0009545700, 0.0010498981, 
    0.0113839157, 0.0120713458, 0.0150390929, 0.0045380464, 0.0090216058, 0.0100194912, 0.0134224489, 0.0141499280, 0.0154726817, 0.0046160813, 0.0001460937, 
    0.0028159609, 0.0023485310, 0.0131379167, 0.0077512142, 0.0082149093, 0.0094532890, 0.0091168997, 0.0162273827, 0.0152643834, 0.0071902420, 0.0091178775 
};

enum mode_t {NAIVE, CPU, CGRA, GPU};

void usage() {
    printf("./convolution_benchmark MODE PATH_TO_INPUT_IMAGE\n");
    printf("\tMODE: naive, cpu, cgra, gpu\n");
    printf("\tPATH_TO_INPUT_IMAGE: *.pgm, *.{jpg,jpeg,JPG,JPEG}\n");
}

int main(int argc, char** argv) {
    if(argc != 3) {
        fprintf(stderr, "no mode/path to input image provided!\n");
        usage();
        return 1;
    }

    // check if naive of opt was chosen
    mode_t mode;

    if(strcmp(argv[1], "naive") == 0) {
        mode = NAIVE;
    } else if(strcmp(argv[1], "cpu") == 0) {
        mode = CPU;
    } else if(strcmp(argv[1], "cgra") == 0) {
        mode = CGRA;
    } else if(strcmp(argv[1], "gpu") == 0) {
        mode = GPU;
    }

    // check if input image is pgm or jpeg
    uint8_t jpg;
    char * extension;

    extension = strrchr(argv[2], '.');
    
    if(strcmp(extension+1, "pgm") == 0) {
        jpg = 0;
    } else if(strcmp(extension+1, "jpg") == 0 || strcmp(extension+1, "jpeg") == 0 || strcmp(extension+1, "JPG") == 0 || strcmp(extension+1, "JPEG") == 0) {
        jpg = 1;
    }

    uint16_t height, width;
    fp_t* input_image;
  

    if(jpg == 0) {
        fp_t* input_image_pgm;
        if(read_pgm(&input_image_pgm, argv[2], 0, 0.0, 1.0, &height, &width) != 0) {
            fprintf(stderr, "could not read pgm image '%s'!\n", argv[1]);
            return 1;
        }
        input_image = input_image_pgm;
    } else if(jpg == 1) {
        fp_t** input_image_jpg;
        if(read_jpeg(&input_image_jpg, argv[2], 0.0, 1.0, &height, &width) != 0) {
            fprintf(stderr, "could not read jpeg image '%s'!\n", argv[1]);
            return 1;
        }
        input_image = input_image_jpg[0];
        free(input_image_jpg[1]);
        free(input_image_jpg[2]);
    }


    fp_t* output_image = (fp_t*) malloc((height-2*KERNEL_CROP)*(width-2*KERNEL_CROP)*sizeof(fp_t));

    if(mode == NAIVE) {
        convolution2d_naive(input_image, height, width, output_image, kernel, KERNEL_SIZE, 0.5);
    } else if(mode == CPU) {
        convolution2d_cpu(input_image, height, width, output_image, kernel, KERNEL_SIZE, 0.5);
    }

    #ifdef DEBUG
    write_pgm(output_image, (height-2*KERNEL_CROP), (width-2*KERNEL_CROP), "output.pgm");
    #endif

    free(output_image);

    return 0;
}
