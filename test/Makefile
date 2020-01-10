CC = gcc
CFLAGS = -Wall -g3
LDFLAGS = -L../pico-cnn
LD_LIBS = -lpico-cnn -lm

TEST_SRCS = utility_functions.c \
			utility_functions.h \
			layers/test_pooling.c \
			layers/test_pooling.h \
			layers/test_activation_function.c \
			layers/test_activation_function.h \
			layers/test_fully_connected.c \
			layers/test_fully_connected.h \
			layers/test_convolution.c \
			layers/test_convolution.h \
			layers/test_batch_normalization.c \
			layers/test_batch_normalization.h

tests: run_tests.c $(TEST_SRCS) libpico-cnn.a
	$(CC) run_tests.c $(TEST_SRCS) $(CFLAGS) -I.. $(LDFLAGS) -o tests $(LD_LIBS)

run: tests
	./tests

.PHONY: clean
clean:
	rm -f tests

.PHONY: libpico-cnn.a
libpico-cnn.a:
	$(MAKE) -C ../pico-cnn
