# CNN_Toolbox

## Introduction
Here are some useful scripts for CNN implementation. 
These scripts are mainly based on Caffe library.

## Summary
*Please specify the path of caffe in caffe_path.txt first!*

- Model Complexity
This script is for FLOPs (FLoating-point OPerations) & parameters calculation.
You need add the *.prototxt file in this folder, and run:
        
        python FLOPs_and_size.py
Note that we regard the vector multiplication as a floating-point operation. Some other people may think it is two independent operations, ie., multiplication and addition. Hence, their result may be as twice as large as ours. 