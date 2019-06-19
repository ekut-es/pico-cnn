# Pico-CNN

## TL;DR
### Ubuntu
```{bash}
sudo apt install libjpeg-dev
```

### Scientific Linux
```{bash}
sudo yum install libjpeg-devel
```

### LeNet-5
LeNet-5 implementation as proposed by Yann LeCun et. al <a id="cit_LeCun1998">[[LeCun1998]](#LeCun1998)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make caffe_lenet_naive
make caffe_lenet_fixed16
./caffe_lenet_naive PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
./caffe_lenet_fixed16 PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
``` 

### LeNet-5 (ARM)
```{bash}
mkdir build
cd build
cmake -DBUILD_ARM_EXAMPLES_=ON ..
make caffe_lenet_arm_cpu
./caffe_lenet_arm_cpu PATH_TO_MNIST_DATASET ../data/caffe_lenet.weights 
``` 

### AlexNet
AlexNet implementation as proposed by Alex Krizhevsky et. al <a id="cit_Krizhevsky2017">[[Krizhevsky2017]](#Krizhevsky2017)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make ekut_es_alexnet_naive
./ekut_es_alexnet_naive PATH_TO_WEIGHTS_FILE PATH_TO_IMAGENET_MEANS PATH_TO_IMAGENET_LABELS PATH_TO_INPUT_IMAGE
``` 

### AlexNet (ARM)
```{bash}
mkdir build
cd build
cmake -DBUILD_ARM_EXAMPLES=ON ..
make ekut_es_alexnet_arm_cpu
./ekut_es_alexnet_arm_cpu PATH_TO_WEIGHTS_FILE PATH_TO_IMAGENET_MEANS PATH_TO_IMAGENET_LABELS PATH_TO_INPUT_IMAGE
``` 

## Set Compiler
Add the following to your `cmake` command:
```
-DCMAKE_C_COMPILER=[PATH_TO_C_COMPILER] -DCMAKE_CXX_COMPILER=[PATH_TO_CXX_COMPILER]
```

## References
<b id="LeCun1998">[LeCun1998]</b> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. [↩](#cit_LeCun1998)

<b id="Krizhevsky2017">[Krizhevsky2017]</b>  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2017. [↩](#cit_Krizhevsky2017)

