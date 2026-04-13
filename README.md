# ismgcc
ISMGCC (InterStellar Medium Gaussian Component Clustering): Finding gas structures in molecular ISM using Gaussian decomposition and Graph theory

This repository contains the Python code that implements the method described in [ISMGCC: Finding Gas Structures in Molecular Interstellar Medium Using Gaussian Decomposition and Graph Theory](https://doi.org/10.1088/1674-4527/ad849b) ( [arXiv:2409.01181](https://arxiv.org/abs/2409.01181) ).
It is designed to find molecular gas structures from PPV data cubes, especially in the direction with crowded emissions, e.g., the inner galaxy.

Please refer to the [tutorial](https://github.com/Haoran-Feng/ismgcc/blob/main/example/tutorial.ipynb) for code examples. 

## Parameter setting

The parameters of ISMGCC are summarized here, same as Table 1 of the [paper](https://doi.org/10.1088/1674-4527/ad849b).

| Symbol | Recommended Value | Tested Values | Description |
|---|---:|---|---|
| $R_{\mathrm{ap}}$ | $3$ | $3, 5, 7$ | Radii of the apertures for velocity coherence clustering, in units of pixels (Sect. 2.1); |
| $\beta$ | 0.50 | $0.25, 0.50, 0.75, 1.0$ | \texttt{bandwidth} coefficient that controls the MeanShift clustering, smaller value leads to more velocity slices (Sect. 2.1); |
| $\delta$ | 0.5 | $0, 0.5, 0.8$ | Minimal value of $N_{\mathrm{same}} / N_{\mathrm{share}}$ for two Gaussian components having a chance to be coherent with each other (Sect. 2.1); |
| $\eta_0$ | 0 | $0, 3$ | Amplitude-to-noise threshold for a Gaussian component to have $P_{\mathrm{real}}>0$ (Eq. 6, Sect. 2.2); |
| $\eta_1$ | 5 | $3, 5, 7$ | Amplitude-to-noise value for a Gaussian component to have $P_{\mathrm{real}}=0.5$ (Eq. 6, Sect. 2.2); |
| $\epsilon$ | 1.5 | $1.5$ | Maximum spatial separation between two Gaussian components for being spatially connected, in units of pixels (Sect. 2.2); |
| $\gamma$ | 0.01 | $0.001, 0.01, 0.1, 1.0$ | The **resolution** parameter of the [Clauset-Newman-Moore algorithm](https://doi.org/10.1103/PhysRevE.70.066111), controls the community sizes in the graph, smaller values lead to larger communities (Sect. 2.3); |
| $N_{\mathrm{pix,min}}$ | 16 | $16$ | Minimal Number of pixels for a structure to be considered as a valid one (Sect. 2.4). |

The recommended values shown above are based on the MWISP 13CO(1-0) data. 
For spectral lines with larger line width (e.g. HI 21cm line), smaller $\beta$ values are recommended, such as $\beta=0.25$.

## Notes & FAQ

Q: Why does the code stop running when I killed a previous run with `Ctrl + C`?

A: To prevent the cache files from being manipulated by multiple processes,  `*.lock` files will be created and removed by the code. 
But when you have mannually killed the program, the `*.lock` file won't be released properly. 
<strong>The solution is deleteing all `.lock` files in the `./.cache/` directory. </strong>
