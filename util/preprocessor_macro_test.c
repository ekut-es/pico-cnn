#include <stdio.h>

int main() {
#ifdef __arm__
    printf("__arm__\n");
#endif
#ifdef __aarch64__
    printf("__aarch64__\n");
#endif
#ifdef __ARM_ARCH_7__
    printf(" __ARM_ARCH_7__\n");
#endif
#ifdef __ARM_ARCH_7A__
    printf("__ARM_ARCH_7A__\n");
#endif
#ifdef __ARM_ARCH_7R__
    printf("__ARM_ARCH_7R__\n");
#endif
#ifdef __ARM_ARCH_7M__
    printf("__ARM_ARCH_7M__\n");
#endif
#ifdef __ARM_ARCH_7S__
    printf("__ARM_ARCH_7S__\n");
#endif
}
