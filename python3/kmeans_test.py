import unittest
from kmeans import kmeans
from random import gauss
import time
import os
import logging
import pickle



logging.basicConfig(level=logging.INFO)
RUN_PLOT_TESTS = True if os.environ.get("RUN_PLOT_TESTS", 'False') == 'True' else False
if RUN_PLOT_TESTS:
    import matplotlib.pyplot as plt


def generate_random_dataset(n):
    """
    Generates a random dataset of 2D points.

    Parameters
    ----------
    n : int
        The number of points to be generated. Must be at least 2.
    """
    assert n >= 2
    dataset = [0] * n
    for i in range(0, n - 1, 2):
        dataset[i] = (gauss(0.33, 1), gauss(0.33, 1))
        dataset[i + 1] = (gauss(0.66, 1), gauss(0.66, 1))   
    return dataset


def read_dataset(filename):
    fp = open(filename)
    npoints = int(next(fp))
    dataset = [0] * npoints
    for i in range(npoints):
        dataset[i] = tuple(float(x) for x in next(fp).split())
    fp.close()
    return dataset


class Timer:

    def __enter__(self):
        self.__start = time.time()


    def __exit__(self, exType, exValue, exTraceback):
        self.__elapsed = time.time() - self.__start
    

    @property
    def elapsed_time(self):
        return self.__elapsed



class KmeansTestCase(unittest.TestCase):


    def test_kmeans_simple_one_cluster_one_dimension(self):
        """
        Let's try a simple dataset with 1D points and one centre.
        """
        dataset = [
            (1,), (3,)
        ]
        expected_centre = [2]
        centres, assignment = kmeans(dataset, 1)
        self.assertEqual(len(centres), 1)
        self.assertEqual(centres[0], expected_centre)
        self.assertTrue(assignment, [0, 0])


    def test_kmeans_corner_case_1(self):
        """
        If the number of required clusters is equal to the cardinality of the dataset,
        then each data point is the centre of its own cluster.
        """
        dataset = [
            (1, 1), (2, 2)
        ]
        expected_centres = [(1, 1), (2, 2)]
        centres, assignment = kmeans(dataset, 2)
        self.assertEqual(len(centres), 2)
        self.assertEqual(set(tuple(x) for x in centres), set(tuple(x) for x in expected_centres))



class KMeansPlotResultsTestCase(unittest.TestCase):


    @unittest.skipUnless(RUN_PLOT_TESTS, "blocks the execution")
    def test_with_simple_dataset(self):
        D = read_dataset('data/dataset.txt')
        X = [x[0] for x in D]
        Y = [x[1] for x in D]
        plt.figure()
        plt.scatter(X, Y)
        timer = Timer()
        with timer:
            centers, assignment = kmeans(D, 2)

        x_blue = [D[i][0] for i in range(len(D)) if assignment[i] == 0]
        y_blue = [D[i][1] for i in range(len(D)) if assignment[i] == 0]
        x_red = [D[i][0] for i in range(len(D)) if assignment[i] == 1]
        y_red = [D[i][1] for i in range(len(D)) if assignment[i] == 1]

        plt.figure()
        plt.scatter(x_blue, y_blue, c='blue')
        plt.scatter(x_red, y_red, c='red')
        plt.show()


    def test_timer(self):
        D = read_dataset('data/dataset.txt')
        timer = Timer()
        with timer:
            centers, assignment = kmeans(D, 2)
        print("Elapsed time:", timer.elapsed_time)
        print("Centres: ", centers)



if __name__ == "__main__":
    unittest.main()