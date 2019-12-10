CC = gcc
CFLAGS = -Wall -g3
LDFLAGS = -L../pico-cnn
LD_LIBS = -lpico-cnn

TEST_SRCS = layers/test_pooling.c \
			layers/test_pooling.h

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