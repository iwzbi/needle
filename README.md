Needle
==============================================

Needle (**Ne**cessary **E**lements of **D**eep **Le**arning) is a mini deep learning framework proposed by CMU for [teaching](https://dlsyscourse.org/) purposes. It includes the basic components of DL frameworks such as modular programming abstraction, dynamic computational graph, automatic differentiation tool, tensor acceleration backend, etc. This repository contains personal implementations, which are also used for further learning and development.

Install
-------
Needle supports CPU and Nvidia GPU as computation backend. To use them, you need to first build a computation library from source:
```
git clone --recursive https://github.com/wzbitl/needle.git
cd needle
mkdir build
cd build
cmake ..
make -j $(nproc)
```
This will generate the corresponding CPU and GPU (if you have a NV GPU device) library files in the `/python/needle/backend_ndarray` folder.

Then, if you want to continue development based on this repository, you can set the environment variable *PYTHONPATH* to tell python where to find the python library. This helps you get immediate influences when change code and rebuild.

```
export NEEDLE_HOME=/path/to/needle    # for example /home/user/needle
export PYTHONPATH=$NEEDLE_HOME/python:${PYTHONPATH}
```
Or you can install Needle python package by setup.py:
```
cd python
python setup.py install --user
cd ..
```
Examples
--------------
Needle is a mini framework that can actually run on it. You can develop your own DL model based on it and expand it as needed. I have implemented several classic DL models in the `/apps` folder for you to quickly understand the ability of Needle. This includes: 
- Training MLP-Resnet with Mnist dataset
- Training Resnet9 with Cifar-10 dataset
- Training sequence model (RNN or LSTM) with Penn Treebank dataset


Features
------------
### 1. Dynamic Computational Graph
Like pytorch, Needle supports the construction of dynamic computational graph by using imperative Python programming. That means you can use Python as the front-end language to form the model expression, and obtain the intermediate results in the real data-flow calculation. Each tensor in the computational graph can be connected through the `op` and `inputs` members in the `Tensor` class, i.e, data in each tensor can be calculated by `self.op(self.inputs)`. Especially, Needle provides *Lazy Mode*,  which can make the data in the tensor not really calculated until the tensor is called.
### 2. Modular Programming Abstraction
The operation mechanism of DL itself is highly modular, and Needle also provides a modular abstraction. Check the contents of the `/python/needle` folder, which contains the following components.
![modular](./figures/modular.png)
### 3. Automatic Differentiation Tool
Needle adopts reverse mode automatic differentiation to obtain the gradient of each tensor in the computational graph. When calling `loss.backwark()`, Needle first builds a topological sort with `loss` tensor as the end, and then get the gradients from back to front according to the chain derivation rule. Through the `gradient` method of `op` member in the  `Tensor` class, you can obtain the "part-gradient" of each input tensor pointing to the current tensor. Add up all the "part-gradient", then you can finally get the gradient value of a tensor relative to total losses in reverse mode.
### 4. Tensor Acceleration Backend
Needle provides Numpy_CPU, CPU and GPU as tensor computation backend. By abstracting the underlying data in the tensor as `NDArray` instead of directly storing the data in memory, Needle has the ability to manipulate and transform the tensor more flexibly. For example, By changing the *stride*, *shape* and *offset* of ndarray, you can obtain different tensor expressions without changing the underlying data layout (unless necessary). For CPU and CPU backend, Needle makes use of tiling and vectorization techniques to better speed up *matmul*, which is a basic computation-intensive operation, and takes advantage of CPU and GPU multi-core parallel processing.

### 5. DL library integration
Needle also gets performance improvement by integrating OneDNN and CuDNN into it. I implemented 4 kinds of *Conv* OP computing (i.e. default CuDNN, Cuda, default CPU, OneDNN). To make OneDNN effective, Set USE_ DNNL is "ON" in `root_dir/CMakeLists.txt`


Acknowledgement
---------------
Thanks to Professor *Tianqi Chen* and Professor *Zico Kolter* and all the TAs for the wonderful lessons. 
