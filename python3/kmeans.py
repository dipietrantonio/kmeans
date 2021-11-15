from random import choice
from math import sqrt, ceil
from statistics import mean
import logging
from mpi4py import MPI
from itertools import chain


def pick_random_centers(points, K):
    """
    Returns ``K`` random points drawn from ``points``.

    Parameters
    ----------
    points : list of tuples
        Dataset where datapoints are drawn from.
    K : int
        Number of points to randomly pick.
    """
    # centers = set()
    # while len(centers) < K:
    #     centers.add(choice(points))
    # return list(centers)
    return points[:K]



def distance(p1, p2):
    """
    Computes the euclidian distance between points ``p1`` and ``p2``.
    """
    return sqrt(sum((c1 - c2)**2 for c1, c2 in zip(p1, p2)))



def kmeans(points, K):
    """
    Partition the dataset ``points`` into ``K`` clusters such that the
    average distance from a point to its cluster center is minimized.

    Parameters
    ----------
    points : list of tuples
        List of data points to cluster.
    K : int
        Number of clusters the dataset will be partitioned into.

    Returns
    -------
    centers : list
        List of points representing the K centers found.
    assignment : list
        An array of integers, varying from ``0`` to ``K-1``, such that the
        value ``k`` at the ``i``-th position indicates that the ``i``-th data
        point belongs to cluster ``k``. 
    """
    comm = MPI.COMM_WORLD
    prank = comm.Get_rank()
    comm_size = comm.Get_size()
    # Each process must start with the same set of (random) centers
    if prank == 0:
        logging.info("Running kmeans..")
        centers = pick_random_centers(points, K)
    else:
        centers = None
    centers = comm.bcast(centers, root=0)
    # Then, each process selects a portion of the datataset to work on
    n_points = len(points)
    n_local_points = ceil(n_points / comm_size)
    local_points = points[prank * n_local_points : (prank + 1) * n_local_points]
    assignment = [-1] * len(local_points)
    n_interations = 0
    while True:
        locally_converged = True
        n_interations += 1
        for j, p in enumerate(local_points):
            closest_center = 0
            closest_center_distance = distance(p, centers[0])
            for i in range(1, len(centers)):
                c = centers[i]
                d = distance(p, c)
                if d < closest_center_distance:
                    closest_center_distance = d
                    closest_center = i
            if assignment[j] != closest_center:
                assignment[j] = closest_center
                locally_converged = False
        
        if all(comm.allgather(locally_converged)):
            if prank == 0:
                logging.info("kmeans ended with {} iterations.".format(n_interations))
            return centers, list(chain(*comm.allgather(assignment)))

        counters = [0] * len(centers)
        centers = [[0] * len(centers[0]) for _ in centers]
        for i, p in enumerate(local_points):
            c = assignment[i]
            for d in range(len(p)):
                centers[c][d] += p[d]
            counters[c] += 1
        # At this point, collect centers and counters from other processes and redistribute
        all_centers_counters = comm.allgather((centers, counters))
        for other_centers, other_counters in (x for i, x in enumerate(all_centers_counters) if i != prank):
            for c in range(len(other_centers)):
                counters[c] += other_counters[c]
                for d, comp in enumerate(other_centers[c]):
                    centers[c][d] += comp    
        for c in range(len(centers)):
            centers[c] = [x/counters[c] for x in centers[c]]