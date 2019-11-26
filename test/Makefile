CC = gcc
LDFLAGS = -L../pico-cnn
LD_LIBS = -lpico-cnn

TEST_SRCS = layers/test_pooling.c \
			layers/test_pooling.h

tests: run_tests.c $(TEST_SRCS)
	$(CC) run_tests.c $(TEST_SRCS) -I.. $(LDFLAGS) -o tests $(LD_LIBS)

run: tests
	./tests

.PHONY: clean
clean:
	rm -f tests
