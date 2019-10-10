

#TODO: check whether -c and -o need to be added here

CC = gcc
CFLAGS = -Wall -g3
LDFLAGS = -lm -ljpeg

# can be changed by calling make DRIVER=driver_ARM
DRIVER = driver_x86

# create directory structure
makedirs:
	mkdir -p $(@D)/lib
	mkdir -p $(@D)/lib/io
	mkdir -p $(@D)/lib/driver
	mkdir -p $(@D)/lib/layers

# remove the library directory
clean:
	rm -rf ./lib

# create the pico-cnn library 
library: makedirs $(DRIVER) io layers
	ar -rcs $(@D)/lib/libpico-cnn.a $(@D)/lib/driver/*.o $(@D)/lib/io/*.o $(@D)/lib/layers/*.o

#---------------------------------------------- driver ----------------------------------------------

# ------------------------- x86 -------------------------
fixed16.o: parameters.h ./driver/fixed16.h ./driver/fixed16.c
	$(CC) driver/fixed16.c $(CFLAGS) -c -o lib/driver/fixed16.o

driver_x86: fixed16.o

# ------------------------- ARM -------------------------
neon_mathfun.o: ./driver/neon_mathfun.h ./driver/neon_mathfun.c
	$(CC) ./driver/neon_mathfun.c $(CFLAGS) -c -o ./lib/driver/neon_mathfun.o

driver_ARM: neon_mathfun.o

#---------------------------------------------- io ----------------------------------------------

# list of all files to consider in io
IO_LIST = read_binary_sample_data.o \
            read_binary_weights.o \
  					read_imagenet_labels.o \
  					read_imagenet_validation_labels.o \
            read_jpeg.o \
  					read_means.o \
  					read_mnist.o \
    				read_pgm.o \
    				read_weights.o \
    				write_float.o \
    				write_pgm.o

# compile all .c files into .o files, write the files to lib/io
$(IO_LIST) : %.o: io/%.c io/%.h
	$(CC) $< $(CFLAGS) -c -o lib/io/$@

io : $(IO_LIST) parameters.h

#---------------------------------------------- layers ----------------------------------------------

# list of all files to consider in layers
LAYERS_LIST = activation_function.o \
              convolution.o \
							fully_connected.o \
							pooling.o

# compile all .c files into .o files, write the files to lib/layers
$(LAYERS_LIST) : %.o: layers/%.c layers/%.h
	$(CC) $< $(CFLAGS) -c -o lib/layers/$@

layers: $(LAYERS_LIST) $(DRIVER) parameters.h
