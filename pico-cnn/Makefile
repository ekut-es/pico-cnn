CC = gcc
CFLAGS = -Wall -g3 -DINFO
LDFLAGS = -lm

libpico-cnn.a: utils io layers
	ar -rcs libpico-cnn.a utils.o io/*.o layers/*.o

# remove the library directory
.PHONY: clean
clean:
	rm -f libpico-cnn.a ./utils.o io/*.o layers/*.o

#---------------------------------------------- utils -----------------------------------------------

utils: parameters.h ./utils.h ./utils.c
	$(CC) utils.c $(CFLAGS) -c -o utils.o

#---------------------------------------------- io ----------------------------------------------

# list of all files to consider in io
IO_SRC = io/read_binary_reference_data.o \
         io/read_binary_weights.o \
  		 io/read_imagenet_labels.o \
  		 io/read_imagenet_validation_labels.o \
         io/read_jpeg.o \
  		 io/read_means.o \
  		 io/read_mnist.o \
    	 io/read_pgm.o \
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
			 layers/pooling.c \
			 layers/batch_normalization.c \
			 layers/concatenate.c
LAYERS_H = $(LAYERS_SRC:.c=.h)
LAYERS_OBJ = $(LAYERS_SRC:.c=.o)

# compile all .c files into .o files, write the files to lib/layers
$(LAYERS_OBJ) : %.o: %.c %.h
	$(CC) $< $(CFLAGS) -c -o $@

layers: $(LAYERS_OBJ) $(DRIVER) parameters.h
