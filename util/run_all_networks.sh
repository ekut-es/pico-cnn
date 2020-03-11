#!/bin/bash

pushd ../onnx_import
source venv/bin/activate

echo "LeNet..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/lenet/lenet.onnx
pushd generated_code/lenet
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/lenet/input.data /nfs/es-genial/pico-cnn/data/onnx/lenet/output.data
popd
echo "LeNet done."
echo ""

echo "LeNet AVG Pooling..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/lenet_avg/lenet_avg.onnx
pushd generated_code/lenet_avg
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/lenet_avg/input.data /nfs/es-genial/pico-cnn/data/onnx/lenet_avg/output.data
popd
echo "LeNet AVG Pooling done."
echo ""

echo "AlexNet..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/bvlc_alexnet/alexnet.onnx
pushd generated_code/alexnet
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/bvlc_alexnet/test_data_set_0/input_0.data /nfs/es-genial/pico-cnn/data/onnx/bvlc_alexnet/test_data_set_0/output_0.data
popd
echo "AlexNet done."
echo ""

echo "CaffeNet..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/bvlc_reference_caffenet/caffenet.onnx
pushd generated_code/caffenet
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/bvlc_reference_caffenet/test_data_set_0/input_0.data /nfs/es-genial/pico-cnn/data/onnx/bvlc_reference_caffenet/test_data_set_0/output_0.data
popd
echo "CaffeNet done."
echo ""

echo "VGG-16..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/vgg16/vgg16.onnx
pushd generated_code/vgg16
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/vgg16/test_data_set_0/input_0.data /nfs/es-genial/pico-cnn/data/onnx/vgg16/test_data_set_0/output_0.data
popd
echo "VGG-16 done."
echo ""

echo "VGG-19..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/vgg19/vgg19.onnx
pushd generated_code/vgg19
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/vgg19/test_data_set_0/input_0.data /nfs/es-genial/pico-cnn/data/onnx/vgg19/test_data_set_0/output_0.data
popd
echo "VGG-19 done."
echo ""

echo "MobileNet-V2..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/mobilenetv2-1.0/mobilenetv2-1.0.onnx
pushd generated_code/mobilenetv2-1
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/mobilenetv2-1.0/test_data_set_0/input_0.data /nfs/es-genial/pico-cnn/data/onnx/mobilenetv2-1.0/test_data_set_0/output_0.data
popd
echo "MobileNet-V2 done."
echo ""

echo "Inception-V3..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/inception_v3/inceptionv3.onnx
pushd generated_code/inceptionv3
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/inception_v3/inceptionv3_input.data /nfs/es-genial/pico-cnn/data/onnx/inception_v3/inceptionv3_output.data
popd
echo "Inception-V3 done."
echo ""

echo "Inception-Resnet-V2..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/inception_resnet_v2/inception_resnet_v2.onnx
pushd generated_code/inception_resnet_v2
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/inception_resnet_v2/inception_resnet_v2_input.data /nfs/es-genial/pico-cnn/data/onnx/inception_resnet_v2/inception_resnet_v2_output.data
popd
echo "Inception-Resnet-V2 done."
echo ""

echo "tc-res8-update..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/tc-res8-update/tc-res8-update.onnx
pushd generated_code/tc-res8-update
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/tc-res8-update/random_tc-res8-update_input.data /nfs/es-genial/pico-cnn/data/onnx/tc-res8-update/random_tc-res8-update_output.data
popd
echo "tc-res8-update done."
echo ""

echo "ekut-raw-cnn3-relu..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu/ekut-raw-cnn3-relu.onnx
pushd generated_code/ekut-raw-cnn3-relu
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu/ekut-raw-cnn3-relu_input.data /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu/ekut-raw-cnn3-relu_output.data
popd
echo "ekut-raw-cnn3-relu done."
echo ""

echo "ekut-raw-cnn3-relu-2..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu-2/ekut-raw-cnn3-relu-2.onnx
pushd generated_code/ekut-raw-cnn3-relu-2
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu-2/ekut-raw-cnn3-relu-2_input.data /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn3-relu-2/ekut-raw-cnn3-relu-2_output.data
popd
echo "ekut-raw-cnn3-relu-2 done."
echo ""

echo "ekut-raw-cnn6-relu..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu/ekut-raw-cnn6-relu.onnx
pushd generated_code/ekut-raw-cnn6-relu
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu/ekut-raw-cnn6-relu_input.data /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu/ekut-raw-cnn6-relu_output.data
popd
echo "ekut-raw-cnn6-relu done."
echo ""

echo "ekut-raw-cnn6-relu-simple..."
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu-simple/ekut-raw-cnn6-relu-simple.onnx
pushd generated_code/ekut-raw-cnn6-relu-simple
make reference_input
./reference_input network.weights.bin /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu-simple/ekut-raw-cnn6-relu-simple_input.data /nfs/es-genial/pico-cnn/data/onnx/ekut-raw-cnn6-relu-simple/ekut-raw-cnn6-relu-simple_output.data
popd
echo "ekut-raw-cnn6-relu-simple done."
echo ""
