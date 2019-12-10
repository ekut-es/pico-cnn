CC = gcc
CFLAGS = -Wall -g3
LDFLAGS = -lm -ljpeg

# can be changed by calling make DRIVER=driver_ARM
driver_ARM = driver/neon_mathfun.o
driver_x86 = driver/fixed16.o
DRIVER = $(driver_x86)

libpico-cnn.a: $(DRIVER) io layers
	ar -rcs libpico-cnn.a driver/*.o io/*.o layers/*.o

# remove the library directory
.PHONY: clean
clean:
	rm -f libpico-cnn.a driver/*.o io/*.o layers/*.o

#---------------------------------------------- driver ----------------------------------------------

# ------------------------- x86 -------------------------
driver/fixed16.o: parameters.h ./driver/fixed16.h ./driver/fixed16.c
	$(CC) driver/fixed16.c $(CFLAGS) -c -o driver/fixed16.o

# ------------------------- ARM -------------------------
driver/neon_mathfun.o: ./driver/neon_mathfun.h ./driver/neon_mathfun.c
	$(CC) ./driver/neon_mathfun.c $(CFLAGS) -c -o driver/neon_mathfun.o

#---------------------------------------------- io ----------------------------------------------

# list of all files to consider in io
IO_SRC = io/read_binary_sample_data.o \
         io/read_binary_weights.o \
  		 io/read_imagenet_labels.o \
  		 io/read_imagenet_validation_labels.o \
         io/read_jpeg.o \
  		 io/read_means.o \
  		 io/read_mnist.o \
    	 io/read_pgm.o \
    	 io/read_weights.o \
    	 io/write_float.o \
    	 io/write_pgm.o
IO_H = $(IO_SRC:.c=.h)
IO_OBJ = $(IO_SRC:.c=.o)

# compile all .c files into .o files, write the files to lib/io
$(IO_OBJ) : %.o: %.c %.h
	$(CC) $< $(CFLAGS) -c -o $@

io : $(IO_OBJ) parameters.h

#---------------------------------------------- layers ----------------------------------------------

# list of all files to consider in layers
LAYERS_SRC = layers/activation_function.c \
             layers/convolution.c \
			 layers/fully_connected.c \
			 layers/pooling.c
LAYERS_H = $(LAYERS_SRC:.c=.h)
LAYERS_OBJ = $(LAYERS_SRC:.c=.o)

# compile all .c files into .o files, write the files to lib/layers
$(LAYERS_OBJ) : %.o: %.c %.h
	$(CC) $< $(CFLAGS) -c -o $@

layers: $(LAYERS_OBJ) $(DRIVER) parameters.h
