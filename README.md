# Introduction to Transformers

This repository follows Richard E. Turner's nice paper [An Introduction to Transformers](https://arxiv.org/pdf/2304.10557) and provides a Python/PyTorch implementation of the equations and architecture. For the most crucial parts, there is a version using loops and another using tensor operations.

This is the general transformer architecture outlined in the paper, without positional embeddings or the specialized auto-regressive language modeling or image clasification.

Limitations:

- Training is omitted from this implementation (although it should be easy to add this)
- There is no batch dimension to make the code easier to follow along
- Some parts of the code are 'unoptimized' on purpose to make it easier to read/map to the paper
