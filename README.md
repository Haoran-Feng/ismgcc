# ismgcc
ISMGCC (InterStellar Medium Gaussian Component Clustering): Finding gas structures in molecular ISM using Gaussian decomposition and Graph theory

This repository contains the Python code that implements the method described in [arXiv:2409.01181](https://arxiv.org/abs/2409.01181).
It is designed to find molecular gas structures from PPV data cubes, especially in the direction with crowded emissions, e.g., the inner galaxy.

The document is currently under construction.
Before its completion, please refer to the [tutorial](https://github.com/Haoran-Feng/ismgcc/blob/main/example/tutorial.ipynb). 

## Notes & FAQ

Q: Why does the code stop running when I killed a previous run with `Ctrl + C`?

A: To prevent the cache files from being manipulated by multiple processes,  `*.lock` files will be created and removed by the code. 
But when you have mannually killed the program, the `*.lock` file won't be released properly. 
**The solution is deleteing all `*.lock` files in the `./.cache/` directory. **
