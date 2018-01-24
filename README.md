# Pico CNN

## TL;DR
LeNet-5 implementation as proposed by Yann LeCun <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
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

## How to get Weights from Caffe
```
cd utils
python convert_caffe_to_pico_cnn.py PATH_TO_TRAIN_PROTOTXT PATH_TO_CAFFEMODEL PATH_TO_WEIGHTS 
``` 
The weights for pico-cnn are now stored in `PATH_TO_WEIGHTS` 


## References
<b id="LeCun1998">[LeCun1998]</b> LeCun, Yann, et al. "Gradient-based learning applied to document recognition." *Proceedings of the IEEE* 86.11 (1998): 2278-2324.[↩](#cit_LeCun1998)
