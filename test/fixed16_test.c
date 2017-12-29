/** 
 * @brief test for fixed16 functions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define FIXED16
#include "pico-cnn/pico-cnn.h"
#include <float.h>
#include <math.h>

#ifdef __aarch64__
#include "arm_neon.h"    
#endif 


int main(int argc, char** argv) {
    
    fp_t a_float =  3.2426;
    fp_t b_float =  5.0167;

    fixed16_t a_fixed16;
    fixed16_t b_fixed16;

    printf("add\n");

    a_float =  3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f + %f =\t%f;\t%f\n", a_float, b_float, a_float + b_float, fixed16_to_float(add_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f + %f =\t%f;\t%f\n", a_float, b_float, a_float + b_float, fixed16_to_float(add_fixed16(a_fixed16, b_fixed16)));

    a_float =  3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f + %f =\t%f;\t%f\n", a_float, b_float, a_float + b_float, fixed16_to_float(add_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f + %f =\t%f;\t%f\n", a_float, b_float, a_float + b_float, fixed16_to_float(add_fixed16(a_fixed16, b_fixed16)));


    printf("sub\n");

    a_float =  3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f - %f =\t%f;\t%f\n", a_float, b_float, a_float - b_float, fixed16_to_float(sub_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f - %f =\t%f;\t%f\n", a_float, b_float, a_float - b_float, fixed16_to_float(sub_fixed16(a_fixed16, b_fixed16)));

    a_float =  3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f - %f =\t%f;\t%f\n", a_float, b_float, a_float - b_float, fixed16_to_float(sub_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f - %f =\t%f;\t%f\n", a_float, b_float, a_float - b_float, fixed16_to_float(sub_fixed16(a_fixed16, b_fixed16)));


    printf("mul\n");

    a_float =  3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f * %f =\t%f;\t%f\n", a_float, b_float, a_float * b_float, fixed16_to_float(mul_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f * %f =\t%f;\t%f\n", a_float, b_float, a_float * b_float, fixed16_to_float(mul_fixed16(a_fixed16, b_fixed16)));

    a_float =  3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f * %f =\t%f;\t%f\n", a_float, b_float, a_float * b_float, fixed16_to_float(mul_fixed16(a_fixed16, b_fixed16)));
    
    a_float =  -3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f * %f =\t%f;\t%f\n", a_float, b_float, a_float * b_float, fixed16_to_float(mul_fixed16(a_fixed16, b_fixed16)));


    printf("div\n");

    a_float =  3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f / %f =\t%f;\t%f\n", a_float, b_float, a_float / b_float, fixed16_to_float(div_fixed16(a_fixed16, b_fixed16)));

    a_float =  -3.2426;
    b_float =  5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f / %f =\t%f;\t%f\n", a_float, b_float, a_float / b_float, fixed16_to_float(div_fixed16(a_fixed16, b_fixed16)));

    a_float =  3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f / %f =\t%f;\t%f\n", a_float, b_float, a_float / b_float, fixed16_to_float(div_fixed16(a_fixed16, b_fixed16)));
    
    a_float =  -3.2426;
    b_float =  -5.0167;
    a_fixed16 = float_to_fixed16(a_float);
    b_fixed16 = float_to_fixed16(b_float);

    printf("%f / %f =\t%f;\t%f\n", a_float, b_float, a_float / b_float, fixed16_to_float(div_fixed16(a_fixed16, b_fixed16)));

    #ifdef __aarch64__
    int i;

    fixed16_t aarr_fixed16[4];
    fixed16_t barr_fixed16[4];
    fixed16_t carr_fixed16[4];
    fixed16_t darr_fixed16[4];

    fixed16x4_t av_fixed16;    
    fixed16x4_t bv_fixed16;    
    fixed16x4_t cv_fixed16; 
    fixed16x4_t dv_fixed16; 

    aarr_fixed16[0] = float_to_fixed16(-4.2515);
    aarr_fixed16[1] = float_to_fixed16(2.7914);
    aarr_fixed16[2] = float_to_fixed16(-1.2352);
    aarr_fixed16[3] = float_to_fixed16(3.0115);

    barr_fixed16[0] = float_to_fixed16(1.2515);
    barr_fixed16[1] = float_to_fixed16(4.7914);
    barr_fixed16[2] = float_to_fixed16(-3.2352);
    barr_fixed16[3] = float_to_fixed16(-2.0115);

    av_fixed16 = vld1_s16(aarr_fixed16);
    bv_fixed16 = vld1_s16(barr_fixed16);
  
    printf("vadd\n");

    cv_fixed16 = vadd_s16(av_fixed16, bv_fixed16);
    vst1_s16(carr_fixed16, cv_fixed16);

    for(i = 0; i < 4; i++) {
        printf("%f + %f = %f\n", fixed16_to_float(aarr_fixed16[i]), fixed16_to_float(barr_fixed16[i]), fixed16_to_float(carr_fixed16[i]));
    }

    printf("vmul\n");

    cv_fixed16 = vmul_fixed16(av_fixed16, bv_fixed16);
    vst1_s16(carr_fixed16, cv_fixed16);

    for(i = 0; i < 4; i++) {
        printf("%f * %f = %f\n", fixed16_to_float(aarr_fixed16[i]), fixed16_to_float(barr_fixed16[i]), fixed16_to_float(carr_fixed16[i]));
    }

    printf("vmla\n");

    dv_fixed16 = vmla_fixed16(cv_fixed16, av_fixed16, bv_fixed16);
    vst1_s16(darr_fixed16, dv_fixed16);

    for(i = 0; i < 4; i++) {
        printf("%f * %f + %f = %f\n", fixed16_to_float(aarr_fixed16[i]), fixed16_to_float(barr_fixed16[i]), fixed16_to_float(carr_fixed16[i]), fixed16_to_float(darr_fixed16[i]));
    }

    printf("vaddv\n");
    printf("%f = %f + %f + %f + %f\n", fixed16_to_float(vaddv_s16(av_fixed16)), fixed16_to_float(aarr_fixed16[0]), fixed16_to_float(aarr_fixed16[1]), fixed16_to_float(aarr_fixed16[2]), fixed16_to_float(aarr_fixed16[3]));

    printf("vmax\n");
    fixed16_t aarrq_fixed16[8];
    fixed16_t barrq_fixed16[8];
    fixed16_t carrq_fixed16[8];

    fixed16x8_t avq_fixed16;    
    fixed16x8_t bvq_fixed16;    
    fixed16x8_t cvq_fixed16; 

    fixed16x4_t cvq_fixed16_0; 
    fixed16x4_t cvq_fixed16_1; 

    aarrq_fixed16[0] = float_to_fixed16(26.63937);
    aarrq_fixed16[1] = float_to_fixed16(-14.34847);
    aarrq_fixed16[2] = float_to_fixed16(-17.23117);
    aarrq_fixed16[3] = float_to_fixed16(26.92824);
    aarrq_fixed16[4] = float_to_fixed16(-24.45074);
    aarrq_fixed16[5] = float_to_fixed16(-27.28157);
    aarrq_fixed16[6] = float_to_fixed16(-26.42929);
    aarrq_fixed16[7] = float_to_fixed16(24.35366);

    barrq_fixed16[0] = float_to_fixed16(-21.946808);
    barrq_fixed16[1] = float_to_fixed16(21.282504);
    barrq_fixed16[2] = float_to_fixed16(21.159183);
    barrq_fixed16[3] = float_to_fixed16(9.563920);
    barrq_fixed16[4] = float_to_fixed16(-21.986626);
    barrq_fixed16[5] = float_to_fixed16(-18.352482);
    barrq_fixed16[6] = float_to_fixed16(-27.299869);
    barrq_fixed16[7] = float_to_fixed16(7.459852);

    avq_fixed16 = vld1q_s16(aarrq_fixed16);
    bvq_fixed16 = vld1q_s16(barrq_fixed16);

    cvq_fixed16 = vmaxq_s16(avq_fixed16, bvq_fixed16);

    vst1q_s16(carrq_fixed16, cvq_fixed16); 


    for(i = 0; i < 8; i++) {
        printf("MAX(%f,%f) = %f\n", fixed16_to_float(aarrq_fixed16[i]), fixed16_to_float(barrq_fixed16[i]), fixed16_to_float(carrq_fixed16[i]));
    }

    printf("folding vmax\n");
    
    cvq_fixed16_0 = vget_low_s16(cvq_fixed16);
    cvq_fixed16_1 = vget_high_s16(cvq_fixed16);

    dv_fixed16 = vpmax_s16(cvq_fixed16_0, cvq_fixed16_0); 
    
    vst1_s16(darr_fixed16, dv_fixed16);
    vst1_s16(aarr_fixed16, cvq_fixed16_0);
    vst1_s16(barr_fixed16, cvq_fixed16_1);

    printf("\n");
    printf("MAX(%f, %f) = %f\n", fixed16_to_float(aarr_fixed16[0]), fixed16_to_float(aarr_fixed16[1]), fixed16_to_float(darr_fixed16[0]));
    printf("MAX(%f, %f) = %f\n", fixed16_to_float(aarr_fixed16[2]), fixed16_to_float(aarr_fixed16[3]), fixed16_to_float(darr_fixed16[1]));

    dv_fixed16 = vpmax_s16(cvq_fixed16_1, cvq_fixed16_1); 

    vst1_s16(darr_fixed16, dv_fixed16);
    vst1_s16(aarr_fixed16, cvq_fixed16_0);
    vst1_s16(barr_fixed16, cvq_fixed16_1);

    printf("\n");
    printf("MAX(%f, %f) = %f\n", fixed16_to_float(aarr_fixed16[0]), fixed16_to_float(aarr_fixed16[1]), fixed16_to_float(darr_fixed16[0]));
    printf("MAX(%f, %f) = %f\n", fixed16_to_float(aarr_fixed16[2]), fixed16_to_float(aarr_fixed16[3]), fixed16_to_float(darr_fixed16[1]));

    #endif 

    return 0;
}
