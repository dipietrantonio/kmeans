# kmeans-python

A Python implementation of K-means.

## Introduction

As part of the materials developed to support Pawsey Supercomputing Centre's training
activities, K-Means is selected as algorithm in the domain of Machine Learning as an 
example of how it is possible to use multiple nodes in a cluster to solve a problem faster.

## Requirements

- `mpi4py`
- `matplotlib` if running tests that show plots.


## Results

Using `mpi4py` allows kmeans to scale very well on a large cluster. What follows are plots describing key performance measures.

![execution time](output/mpi4py/exec_time.png "Execution times")

![speedup](output/mpi4py/speedup.png "Speedup")

![efficiency](output/mpi4py/efficiency.png "Efficiency")
