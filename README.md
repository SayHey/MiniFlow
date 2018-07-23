# MiniFlow

The goal of this personal self-educational project is to create a dataflow framework for working with computational graphs, designing and training of neural networks, similar to simplified TensorFlow framework from google. Written on \Cpp17. Planning to port tensor math library to CUDA in the near future.

## Project architecture:

* **Node.h** contains code of different computational graph nodes (layers on neural network)
* **Graph.h** contains computational graph interface such as training and predicting fuctions
* **DynamicTensor.h** and **StaticTensor.h** are defferent tensor math libraries. 
  DynamicTensor data container is based on std::vector, while StaticTensor is based on std::array.
