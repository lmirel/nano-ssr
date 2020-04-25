# nano-ssr
speed limit sign recognition with Jetson Nano

follow the info from here to transfer learning of a network (I used resnet18) using the german traffic sign set (for example) 
https://github.com/dusty-nv/jetson-inference

german traffic sign dataset is here https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html

convert to onxx and rename to resnet18_e34.onnx, the labels.txt file should be available as well.

start with: ./tsr-camera.py --model=resnet18_e34.onnx --input_blob=input_0 --output_blob=output_0 --labels=labels.txt

use at your own risk and don't blame me for your speeding tickets

contact me if you want to use my trained model, I get over 99% accuracy.

donate if you feel this helped you: https://paypal.me/lazarmirel
