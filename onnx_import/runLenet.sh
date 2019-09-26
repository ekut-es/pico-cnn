cd /local/weinnils/git/pico-cnn/onnx_import 
python3.6 onnx_to_pico_cnn.py --input /nfs/es-genial/pico-cnn/data/onnx/lenet.onnx
cd generated_code/lenet
make lenet
./lenet /local/weinnils/git/mnist network.weights.bin 
