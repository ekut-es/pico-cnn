# Pico CNN

## TL;DR
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

### AlexNet
AlexNet implementation as proposed by Alex Krizhevsky et. al <a id="cit_Krizhevsky2017">[[Krizhevsky2017]](#Krizhevsky2017)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make ekut_es_alexnet_naive
``` 

### VGG
VGG implementation as proposed by Karen Simonyan et. al <a id="cit_Simonyan2014">[[Simonyan2014]](#Simonyan2014)</a>
```{bash}
mkdir build
cd build
cmake -DBUILD_EXAMPLES=ON ..
make ekut_es_vgg_naive
``` 


## Set Compiler
Add the following to your `cmake` command:
```
-DCMAKE_C_COMPILER=[PATH_TO_C_COMPILER] -DCMAKE_CXX_COMPILER=[PATH_TO_CXX_COMPILER]

```

## References
<b id="LeCun1998">[LeCun1998]</b> Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, Nov. 1998. [↩](#cit_LeCun1998)
<b id="Krizhevsky2017">[Krizhevsky2017]</b>  A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” Communications of the ACM, vol. 60, no. 6, pp. 84–90, May 2017. [↩](#cit_Krizhevsky2017)
 <b id="Simonyan2014">[Simonyan2014]</b> K. Simonyan and A. Zisserman, “Very Deep Convolutional Networks for Large-Scale Image Recognition,” arXiv:1409.1556 [cs], Sep. 2014.  [↩](#cit_Simonyan2014)

