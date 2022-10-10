# Kmeans

A collection of K-Means implementations using various parallel programming frameworks, created for testing purposes.

## Repository structure

Each folder contains a different implementation, each to be treated as an independent codebase. In particular

- `cpp`: a MPI + OpenMP implementation written in C++;
- `cuda`: a pure CUDA implementation (single GPU);
- `hip`: a HIP implementation, the starting point being the CUDA implementation after HIPIFYcation;
- `openmp`: a pure OpenMP implementation in C++;
- `python3`: a Python3 implementation using MPI.


## How to run

The dataset to run the program with must be generated separately using the `generate_dataset.py` Python script.

```
$ python3 generate_dataset.py 
Usage: generate_dataset.py <n points> <filename>
```

A dataset of at least 50 million points is suggested. Tests within the projects read the path to the dataset file
from the `KMEANS_DATASET` environment variable. Here is how you would run the `cpp` implementation:

```
cdipietrantonio@nid00012:/group/pawsey0001/cdipietrantonio/kmeans> python3 generate_dataset.py 50000000 dataset.txt
cdipietrantonio@nid00012:/group/pawsey0001/cdipietrantonio/kmeans> export KMEANS_DATASET=`pwd`/dataset.txt
cdipietrantonio@nid00012:/group/pawsey0001/cdipietrantonio/kmeans> cd cpp/
cdipietrantonio@nid00012:/group/pawsey0001/cdipietrantonio/kmeans/cpp> make
[ -d obj ] || mkdir obj
[ -d bin ] || mkdir bin
CC  -O3 -g  -o bin/tests tests/tests.cpp #-L/software/projects/pawsey0001/cdipietrantonio/kmeans/cpp/obj -lmap-sampler-pmpi -lmap-sampler -Wl,--eh-frame-hdr -Wl,-rpath=/software/projects/pawsey0001/cdipietrantonio/kmeans/cpp/obj -lmpi

[...]
cdipietrantonio@nid00012:/group/pawsey0001/cdipietrantonio/kmeans/cpp> srun ./bin/tests
Rank 0 is running on CPU 0
Kmeans execution time: 122
Center 0: (-0.0849834, -0.0836341), centre 1: (1.07552, 1.07393)
All tests passed.
```