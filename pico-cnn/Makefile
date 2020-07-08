CC = g++
AR = ar
CFLAGS = -std=c++11 -Wall -O2 -march=native -DINFO 
LDFLAGS =

libpico-cnn.a: layers io parameters.h utils.h pico-cnn.h
	$(AR) -rcs libpico-cnn.a *.o layers/*.o layers/activation_functions/*.o layers/pooling/*.o io/*.o

# remove the library directory
.PHONY: clean
clean:
	rm -f libpico-cnn.a *.o layers/*.o layers/activation_functions/*.o layers/pooling/*.o io/*.o

#---------------------------------------------- utils -----------------------------------------------

# utils: parameters.h ./utils.h ./utils.c
# 	$(CC) utils.c $(CFLAGS) -c -o utils.o

#---------------------------------------------- io ----------------------------------------------

# list of all files to consider in io
IO_SRC = io/read_binary_reference_data.cpp \
         io/read_binary_weights.cpp \
         io/read_imagenet_labels.cpp \
         io/read_imagenet_validation_labels.cpp \
         io/read_jpeg.cpp \
         io/read_means.cpp \
         io/read_mnist.cpp \
         io/read_pgm.cpp \
         io/write_float.cpp \
         io/write_pgm.cpp

IO_H = $(IO_SRC:.cpp=.h)
IO_OBJ = $(IO_SRC:.cpp=.o)

# compile all .c files into .o files, write the files to lib/io
$(IO_OBJ) : %.o: %.cpp %.h
	$(CC) $< $(CFLAGS) -c -o $@

io : $(IO_OBJ) parameters.h

#---------------------------------------------- layers ----------------------------------------------

# list of all files to consider in layers
LAYERS_SRC = tensor.cpp \
             layers/layer.cpp \
             layers/convolution.cpp \
             layers/pooling/pooling.cpp \
             layers/pooling/max_pooling.cpp \
             layers/pooling/average_pooling.cpp \
             layers/pooling/global_max_pooling.cpp \
             layers/pooling/global_average_pooling.cpp \
             layers/activation_functions/activation_function.cpp \
             layers/activation_functions/clip.cpp \
             layers/activation_functions/lrn.cpp \
             layers/activation_functions/relu.cpp \
             layers/activation_functions/sigmoid.cpp \
             layers/activation_functions/softmax.cpp \
             layers/activation_functions/tan_h.cpp \
             layers/fully_connected.cpp \
             layers/batch_normalization.cpp

LAYERS_H = $(LAYERS_SRC:.cpp=.h)
LAYERS_OBJ = $(LAYERS_SRC:.cpp=.o)

# compile all .c files into .o files, write the files to lib/layers
$(LAYERS_OBJ) : %.o: %.cpp %.h
	$(CC) $< $(CFLAGS) -c -o $@

layers: $(LAYERS_OBJ) parameters.h utils.h
