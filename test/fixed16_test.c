/** 
 * @brief test for fixed16 functions
 *
 * @author Konstantin Luebeck (University of Tuebingen, Chair for Embedded Systems)
 */

#define FIXED16
#include "pico-cnn/pico-cnn.h"
#include <float.h>
#include <math.h>

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


    return 0;
}
