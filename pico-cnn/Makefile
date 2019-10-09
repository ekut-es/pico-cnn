makedirs:
	mkdir -p $(@D)/lib
	mkdir -p $(@D)/lib/io
	mkdir -p $(@D)/lib/driver
	mkdir -p $(@D)/lib/layers

clean:
	rm -rf ./lib
	rm -f libpico-cnn.a

# driver_x86 or driver_ARM
DRIVER = driver_x86

library: makedirs $(DRIVER) io layers
	ar -rcs $(@D)/libpico-cnn.a $(@D)/lib/driver/*.o $(@D)/lib/io/*.o $(@D)/lib/layers/*.o

#-------------------- driver ----------------------

fixed16.o: parameters.h ./driver/fixed16.h ./driver/fixed16.c
	gcc ./driver/fixed16.c -c -o ./lib/driver/fixed16.o

# shouldn't work as this is not an ARM-PC
neon_mathfun.o: ./driver/neon_mathfun.h ./driver/neon_mathfun.c
	gcc ./driver/neon_mathfun.c -lm -c -o ./lib/driver/neon_mathfun.o

driver_ARM: neon_mathfun.o

driver_x86: fixed16.o

#-------------------- io -------------------------

read_binary_sample_data.o: parameters.h ./io/read_binary_sample_data.c ./io/read_binary_sample_data.h
	gcc ./io/read_binary_sample_data.c -c -o ./lib/io/read_binary_sample_data.o

read_binary_weights.o: parameters.h ./io/read_binary_weights.c ./io/read_binary_weights.h
	gcc ./io/read_binary_weights.c -c -o ./lib/io/read_binary_weights.o

read_imagenet_labels.o: ./io/read_imagenet_labels.c ./io/read_imagenet_labels.h
	gcc ./io/read_imagenet_labels.c -lm -c -o ./lib/io/read_imagenet_labels.o

read_imagenet_validation_labels.o: ./io/read_imagenet_validation_labels.h ./io/read_imagenet_validation_labels.c
	gcc ./io/read_imagenet_validation_labels.c -c -o ./lib/io/read_imagenet_validation_labels.o

read_jpeg.o: ./io/read_jpeg.h ./io/read_jpeg.c
	gcc ./io/read_jpeg.c -c -o ./lib/io/read_jpeg.o

read_means.o: parameters.h ./io/read_means.h ./io/read_means.c
	gcc ./io/read_means.c -c -o ./lib/io/read_means.o

read_mnist.o: parameters.h ./io/read_mnist.c ./io/read_mnist.h
	gcc ./io/read_mnist.c -c -o ./lib/io/read_mnist.o

read_pgm.o: parameters.h ./io/read_pgm.h ./io/read_pgm.c
	gcc ./io/read_pgm.c -c -o ./lib/io/read_pgm.o

read_weights.o: parameters.h ./io/read_weights.h ./io/read_weights.c
	gcc ./io/read_weights.c -c -o ./lib/io/read_weights.o

write_float.o: parameters.h ./io/write_float.h ./io/write_float.c
	gcc ./io/write_float.c -c -o ./lib/io/write_float.o

write_pgm.o: parameters.h ./io/write_pgm.h ./io/write_pgm.c
	gcc ./io/write_pgm.c -c -o ./lib/io/write_pgm.o

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

IO_FILES = ./lib/io/read_binary_sample_data.o \
          ./lib/io/read_binary_weights.o \
					./lib/io/read_imagenet_labels.o \
					./lib/io/read_imagenet_validation_labels.o \
          ./lib/io/read_jpeg.o \
					./lib/io/read_means.o \
					./lib/io/read_mnist.o \
  				./lib/io/read_pgm.o \
  				./lib/io/read_weights.o \
  				./lib/io/write_float.o \
  				./lib/io/write_pgm.o

io: $(IO_LIST)

#---------------- layers -------------------------

activation_function.o: parameters.h ./layers/activation_function.h ./layers/activation_function.c $(DRIVER)
	gcc ./layers/activation_function.c -c -o ./lib/layers/activation_function.o

convolution.o: parameters.h ./layers/convolution.h ./layers/convolution.c $(DRIVER)
	gcc ./layers/convolution.c -c -o ./lib/layers/convolution.o

pooling.o: parameters.h ./layers/pooling.h ./layers/pooling.c $(DRIVER)
	gcc ./layers/pooling.c -c -o ./lib/layers/pooling.o

fully_connected.o: parameters.h ./layers/fully_connected.h ./layers/fully_connected.c $(DRIVER)
	gcc ./layers/fully_connected.c -c -o ./lib/layers/fully_connected.o

layers: activation_function.o convolution.o fully_connected.o pooling.o
