CC = g++
CFLAGS = -std=c++11 -Wall -g3 -DINFO -DDEBUG 
LDFLAGS = -L../pico-cnn
LD_LIBS = -lpico-cnn -lcppunit

TEST_SRCS = layers/test_activation_functions.cpp \
            layers/test_batch_normalization.cpp \
            layers/test_convolution.cpp \
            layers/test_fully_connected.cpp \
            layers/test_pooling.cpp \
            layers/test_tensor.cpp \

tests: main.cpp $(TEST_SRCS) libpico-cnn.a
	$(CC) main.cpp $(TEST_SRCS) $(CFLAGS) -I../../pico-cnn $(LDFLAGS) -o tests $(LD_LIBS)

run: tests
	./tests

.PHONY: clean
clean:
	rm -f tests

.PHONY: libpico-cnn.a
libpico-cnn.a:
	$(MAKE) -C ../pico-cnn
