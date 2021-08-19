#!/usr/bin/env python3
# +======================================================================+
# |                                                                      |
# | File: matrix_multiply.py                                             |
# | Author: Jim Kelly | N9763686                                         |
# | Purposes: This script has been created for the purposes of an        |
# |      oral presentation for CAB401 High Performance and Parallel      |
# |      Computing, Semester 2, 2021, at the Queensland University of    |
# |      Technology (QUT). The script demonstrates a basic comparison    |
# |      between the best sequential matrix multiplication with Python   |
# |      and the parallelised implementation of matrix multiplication    |
# |      using Apache-Spark's BlockMatrix.                               |
# | Description: This script tests the performance of the best           |
# |      sequential matrix multiplication against that of Apache Spark   |
# |      over several different values of N. To change the number of     |
# |      tests performed per N-sized matrix change 'iterations'. To      |
# |      change the values of N simply update the array 'N'.             |
# +======================================================================+
# |                           Imports                                    |
# +======================================================================+

import random
import json
from timeit import default_timer as timer
from numpy import mat
from scipy.sparse import lil_matrix
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import *

# +======================================================================+
# |                         Start of File                                |
# +======================================================================+

# Create and start Spark session
app_name = 'PySpark Matrix Multiplication Example'
master = 'local'
spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define global testing variables
iterations = 10
N_values = [1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Method to create a populated matrix of size N * N, with potential
# values ranging from (max_value - 1) through to max_value.
def create_matrix(size, max_value):
    return [[random.randint((max_value * -1), max_value) for i in range(size)] \
            for j in range(size)]

# Method to create a matrix populated with 0's of size N * N.
def create_empty_matrix(size):
    return [[0 for i in range(size)] for j in range(size)]

# Method to multiply two matrices of the same dimensions, i.e. N * N.
def matrix_multiply(A, B, C, size):
    for i in range(size):
        for j in range(size):
            total = 0 # Initialise total to 0
            for k in range(size):
                total += A[i][k] * B[k][j] # Perform matrix multiply
            C[i][j] = total

    # Return the result of the matrix multiplication
    return C

# Method to convert a Resilient Distributed Dataset (RDD) to a BlockMatrix object.
def as_block_matrix(rdd, rows, columns):
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))
    ).toBlockMatrix(rows, columns)

# Method to convert an indexed row matrix to a local array using Scipy 'lil_matrix'
# operation.
def indexedrowmatrix_to_array(matrix):
    # Create an empty array of the same dimensions as the matrix
    result = lil_matrix((matrix.numRows(), matrix.numCols()))

    # Iterate through each row and set values in the empty array
    for indexed_row in matrix.rows.collect():
        result[indexed_row.index] = indexed_row.vector

    # Return the local array
    return result

# Method to fetch the corresponding matrices for Spark matrix multiplication.
def fetch_matrices(matrix_array, iteration, N):
    for x in range(len(matrix_array)):
        if matrix_array[x]['N'] == N:
            temp = matrix_array[x]['Matrices']
            for y in range(len(temp)):
                if temp[y]['Iteration'] == iteration:
                    return [temp[y]['A'], temp[y]['B']]

# Method to test the sequential matrix multiplication of two randomly generated
# matrices of size N * N. This method performs the test the same amount of times
# as is supplied by 'iterations'.
def test_sequential(iterations, N_values, max_value):
    # Initialise result storage objects
    result_array = {'Name': 'Sequential Tests', 'Results': []}
    matrix_array = []

    # Iterate through all values of N
    for N in N_values:
        print("Running sequential matrix multiply of", N, 'by', N, 'matrices.')

        # Initialise dictionaries to store results of test
        results = {'N': N, 'Execution Time': [0] * iterations}
        matrices = {'N': N, 'Matrices': []}

        # Perform the number of tests provided in 'iterations'
        for i in range(iterations):
            # Initialise matrices
            A = create_matrix(N, max_value)
            B = create_matrix(N, max_value)
            C = create_empty_matrix(N)

            # Perform and time matrix multiplication
            start = timer() #
            C = matrix_multiply(A, B, C, N) #
            end = timer() #

            # Store results
            results['Execution Time'][i] = end - start
            matrices['Matrices'].append({'Iteration': i, 'A': A, 'B': B})

        # Add results to final output
        result_array['Results'].append(results)
        matrix_array.append(matrices)
        print("Completed testing sequential matrix multiply of", N, 'by', N, 'matrices.')

    # Return test results
    return [result_array, matrix_array]

# Method to test the Spark matrix multiplication of two randomly generated
# matrices of size N * N. This method performs the test the same amount of times
# as is supplied by 'iterations'.
def test_spark(iterations, N_values, matrix_array):
    # Initialise result storage objects
    result_array = {'Name': 'Spark Tests', 'Results': []}

    # Iterate through all values of N
    for N in N_values:
        print("Running Spark matrix multiply of", N, 'by', N, 'matrices.')

        # Initialise dictionary to store results of test
        results = {'N': N, 'Execution Time': [0] * iterations}

        for i in range(iterations):
            # Fetch matrices
            A, B = fetch_matrices(matrix_array, i, N)

            # Convert arrays to RDDs
            A_rdd = spark.sparkContext.parallelize(A)
            B_rdd = spark.sparkContext.parallelize(B)

            # Perform and time matrix multiplication
            start = timer() #
            C_matrix = as_block_matrix(A_rdd, N, N).multiply(as_block_matrix(B_rdd, N, N)) #
            end = timer() #

            # Store results
            results['Execution Time'][i] = end - start

        # Add results to the final output
        result_array['Results'].append(results)
        print("Completed testing Spark matrix multiply of", N, 'by', N, 'matrices.')

    # Return the test results
    return result_array

# Method to run the testing methods and then write the results to a .json file.
def run_test(iterations, N_values):
    print("Initialising testing suite...")

    # Run tests
    sequential_test, matrices = test_sequential(iterations, N_values, N_values[-1])
    parallel_test = test_spark(iterations, N_values, matrices)

    print("Writing results...")

    # Write results to file
    output = open("test_results.json", "w")
    output.write("[")
    output.write(json.dumps(sequential_test))
    output.write(",")
    output.write(json.dumps(parallel_test))
    output.write("]")
    output.close()
    print("Operation complete.")

# Run the tests
run_test(iterations, N_values)

# +======================================================================+
# |                           End of File                                |
# +======================================================================+