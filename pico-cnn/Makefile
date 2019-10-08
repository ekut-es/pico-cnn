fixed16.o: parameters.h ./driver/fixed16.h ./driver/fixed16.c
	gcc ./driver/fixed16.c -o ./driver/fixed16.o

# shouldn't work as this is not an ARM-PC
neon_mathfun.o: ./driver/neon_mathfun.h ./driver/neon_mathfun.c
	gcc ./driver/neon_mathfun.c -lm -o ./driver/neon_mathfun.o


driver: fixed16.o neon_mathfun.o
	gcc ./driver/fixed16.o ./driver/neon_mathfun.o -o driver.o

#####

read_binary_sample_data.o: parameters.h ./io/read_binary_sample_data.c ./io/read_binary_sample_data.h
	gcc ./io/read_binary_sample_data.c -c -o ./io/read_binary_sample_data.o

read_binary_weights.o: parameters.h ./io/read_binary_weights.c ./io/read_binary_weights.h
	gcc ./io/read_binary_weights.c -c -o ./io/read_binary_weights.o

read_imagenet_labels.o: ./io/read_imagenet_labels.c ./io/read_imagenet_labels.h
	gcc ./io/read_imagenet_labels.c -lm -c -o ./io/read_imagenet_labels.o

read_imagenet_validation_labels.o: ./io/read_imagenet_validation_labels.h ./io/read_imagenet_validation_labels.c
	gcc ./io/read_imagenet_validation_labels.c -c -o ./io/read_imagenet_validation_labels.o

read_jpeg.o: ./io/read_jpeg.h ./io/read_jpeg.c
	gcc ./io/read_jpeg.c -c -o ./io/read_jpeg.o

read_means.o: parameters.h ./io/read_means.h ./io/read_means.c
	gcc ./io/read_means.c -c -o ./io/read_means.o

read_mnist.o: parameters.h ./io/read_mnist.c ./io/read_mnist.h
	gcc ./io/read_mnist.c -c -o ./io/read_mnist.o

read_pgm.o: parameters.h ./io/read_pgm.h ./io/read_pgm.c
	gcc ./io/read_pgm.c -c -o ./io/read_pgm.o

read_weights.o: parameters.h ./io/read_weights.h ./io/read_weights.c
	gcc ./io/read_weights.c -c -o ./io/read_weights.o

write_float.o: parameters.h ./io/write_float.h ./io/write_float.c
	gcc ./io/write_float.c -c -o ./io/write_float.o

write_pgm.o : parameters.h ./io/write_pgm.h ./io/write_pgm.c
	gcc ./io/write_pgm.c -c -o ./io/write_pgm.o

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

IO_FILES = ./io/read_binary_sample_data.o \
          ./io/read_binary_weights.o \
					./io/read_imagenet_labels.o \
					./io/read_imagenet_validation_labels.o \
          ./io/read_jpeg.o \
					./io/read_means.o \
					./io/read_mnist.o \
  				./io/read_pgm.o \
  				./io/read_weights.o \
  				./io/write_float.o \
  				./io/write_pgm.o

io.o: $(IO_LIST)
	gcc $(IO_FILES) -lm -o ./io/io.o
