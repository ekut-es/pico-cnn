# Pico CNN

## TL;DR
LeNet-5 implementation as proposed by Yann LeCun <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>
```{bash}
mkdir build
cd build
cmake ..
make caffe_lenet
./caffe_lenet PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
``` 

## Build Everything
```
mkdir build
cd build
cmake -DBUILD_TESTS=ON -DBUILD_UTILS=ON -DBUILD_EXAMPLES=ON ..
make
```

## How to get a CNN Structure and Weights from Caffe
Clone `tiny-dnn-nets`:
```
git clone git@atreus.informatik.uni-tuebingen.de:luebeck/tiny-dnn-nets.git
```

Follow the `README.md`

The CNN for pico-cnn has to be constructed by hand with the help of `get_weights_file` from `tiny-dnn-nets`

## References
<b id="LeCun1998">[LeCun1998]</b> LeCun, Yann, et al. "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86.11 (1998): 2278-2324.[↩](#cit_LeCun1998)
