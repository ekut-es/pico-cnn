CC = g++
CFLAGS = -Wall -g3 -DINFO -DDEBUG
LDFLAGS = -L../../pico-cnn-cpp
LD_LIBS = -lpico-cnn-cpp -lcppunit

TEST_SRCS = test_tensor.cpp \
            test_activation_functions.cpp \
            test_fully_connected.cpp \
            test_convolution.cpp \

tests: main.cpp $(TEST_SRCS) libpico-cnn-cpp.a
	$(CC) main.cpp $(TEST_SRCS) $(CFLAGS) -I../../pico-cnn-cpp $(LDFLAGS) -o tests $(LD_LIBS)

run: tests
	./tests

.PHONY: clean
clean:
	rm -f tests

.PHONY: libpico-cnn-cpp.a
libpico-cnn-cpp.a:
	$(MAKE) -C ../../pico-cnn-cpp
