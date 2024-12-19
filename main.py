import random

def generate_matrix(dim, points, range_min=-1500.0, range_max=1500.0):
    """
    Generate a matrix of size dim x points with random values in the specified range.

    Args:
        dim (int): Number of dimensions (rows of the matrix).
        points (int): Number of points (columns of the matrix).
        range_min (float): Minimum value for the matrix elements.
        range_max (float): Maximum value for the matrix elements.

    Returns:
        list: A 2D list representing the matrix.
    """
    return [
        [random.uniform(range_min, range_max) for _ in range(points)]
        for _ in range(dim)
    ]

def write_matrix_to_file(matrix, filename):
    """
    Write a 2D matrix to a file.

    Args:
        matrix (list): The 2D list to write to the file.
        filename (str): The name of the file.
    """
    with open(filename, "w") as file:
        for row in matrix:
            file.write(" ".join(map(str, row)) + "\n")

if __name__ == "__main__":
    # User input for dimensions and points

    # Generate the matrix
    matrix = generate_matrix(96, 100_000)

    # Write the matrix to a text file
    filename = "text.txt"
    write_matrix_to_file(matrix, filename)

    #print(f"Matrix of size {dim}x{points} written to {filename}.")
