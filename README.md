# Pico CNN

## TL;DR
LeNet-5 implementation as proposed by Yann LeCun <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make caffe_lenet_naive
make caffe_lenet_fixed16
./caffe_lenet_naive PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
./caffe_lenet_fixed16 PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
``` 

## Build Everything
```
mkdir build
cd build
cmake -DBUILD_UTILS=ON -DBUILD_EXAMPLES=ON ..
make
```

## Set Compiler
Add the following to your `cmake` command:
```
-DCMAKE_C_COMPILER=[PATH_TO_C_COMPILER] -DCMAKE_CXX_COMPILER=[PATH_TO_CXX_COMPILER]

```

## References
<b id="LeCun1998">[LeCun1998]</b> LeCun, Yann, et al. "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86.11 (1998): 2278-2324.[↩](#cit_LeCun1998)
