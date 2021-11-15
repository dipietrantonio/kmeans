from random import gauss



def generate_random_dataset(n):
    """
    Generates a random dataset of 2D points.

    Parameters
    ----------
    n : int
        The number of points to be generated. Must be at least 2.
    """
    assert n >= 2
    for i in range(0, n - 1, 2):
        yield (gauss(0.33, 1), gauss(0.33, 1))
        yield (gauss(0.66, 1), gauss(0.66, 1))


if __name__ == "__main__":
    
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <n points> <filename>")
        exit(0)
    fp = open(f"{sys.argv[2]}", "w") 
    fp.write(f"{sys.argv[1]}\n")
    for point in generate_random_dataset(int(sys.argv[1])):
        fp.write(f"{point[0]} {point[1]}\n")
    fp.close()

