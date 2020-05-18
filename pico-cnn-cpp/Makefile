CC = g++
CFLAGS = -Wall -g3 -DINFO
LDFLAGS =

libpico-cnn-cpp.a: layers io parameters.h utils.h
	ar -rcs libpico-cnn-cpp.a *.o layers/*.o layers/activation_functions/*.o layers/pooling/*.o io/*.o

# remove the library directory
.PHONY: clean
clean:
	rm -f libpico-cnn-cpp.a *.o layers/*.o layers/activation_functions/*.o layers/pooling/*.o io/*.o

#---------------------------------------------- utils -----------------------------------------------

# utils: parameters.h ./utils.h ./utils.c
# 	$(CC) utils.c $(CFLAGS) -c -o utils.o

#---------------------------------------------- io ----------------------------------------------

# list of all files to consider in io
IO_SRC = io/read_binary_reference_data.cpp \
         io/read_binary_weights.cpp

IO_H = $(IO_SRC:.cpp=.h)
IO_OBJ = $(IO_SRC:.cpp=.o)

# compile all .c files into .o files, write the files to lib/io
$(IO_OBJ) : %.o: %.cpp %.h
	$(CC) $< $(CFLAGS) -c -o $@

io : $(IO_OBJ) parameters.h

#---------------------------------------------- layers ----------------------------------------------

# list of all files to consider in layers
LAYERS_SRC = tensor_shape.cpp \
             tensor.cpp \
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
