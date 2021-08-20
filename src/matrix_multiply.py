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
# | Description: To change the matrix dimensions replace the value for   |
# |      'N'. The script will first calculate the sequential values of   |
# |      of two matrices populated with number N randomised values. The  |
# |      script will then compute the same operation using Spark's       |
# |      parallelisation methods, i.e. BlockMatrix.multiply().           |
# +======================================================================+
# | References:                                                          |
# | as_block_matrix: Taken from Stack Overflow,                          |
# |     How to Multiply Two Numpy Matrices in PySpark?,                  |
# | 'https://stackoverflow.com/questions/42889965/multiply-two-numpy-    |
# |     matrices-in-pyspark'                                             |
# | indexrowmatrix_to_array: Taken from Stack Overflow,                  |
# |     How to Convert a DistributedMatrix to Scipy Sparse or            |
# |         Numpy Array,                                                 |
# | 'https://stackoverflow.com/questions/54083978/convert-a-             |
# |     distributedmatrix-to-scipy-sparse-or-numpy-array'                |
# +======================================================================+
# |                           Imports                                    |
# +======================================================================+

import random
from timeit import default_timer as timer
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

# Define global variables
N = 2 # Matrix dimensions

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

# Method to convert a Resilient Distributed Dataset (RDD) to a BlockMatrix object
def as_block_matrix(rdd, rows, columns):
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))
    ).toBlockMatrix(rows, columns)

# Method to convert an indexed row matrix to a local array using Scipy 'lil_matrix'
def indexedrowmatrix_to_array(matrix):
    # Create an empty array of the same dimensions as the matrix
    result = lil_matrix((matrix.numRows(), matrix.numCols()))

    # Iterate through each row and set values in the empty array
    for indexed_row in matrix.rows.collect():
        result[indexed_row.index] = indexed_row.vector

    # Return the local array
    return result

# Initialise matrices
A = create_matrix(N, 500)
B = create_matrix(N, 500)
C = create_empty_matrix(N)

print('Performing standard matrix multiplication')

# Perform and time matrix multiplication
start = timer() #
C = matrix_multiply(A, B, C, N) #
end = timer() #

# Print the execution time
print('Best Sequential execution time:', end - start)

# Convert arrays to RDDs
A_rdd = spark.sparkContext.parallelize(A)
B_rdd = spark.sparkContext.parallelize(B)

# Perform and time matrix multiplication
start = timer() #
C_matrix = as_block_matrix(A_rdd, N, N).multiply(as_block_matrix(B_rdd, N, N)) #
end = timer() #

# Print the execution time
print('Apache Spark execution time:', end - start)

# Convert the resulting BlockMatrix to a local array
result = indexedrowmatrix_to_array(C_matrix.toIndexedRowMatrix())

# Print resulting matrix if it's not excessively large
if N <= 4:
    print("Printing sequential result matrix.")
    for row in C:
        print(row)
    print("Printing Spark result matrix")
    print(result)

# +======================================================================+
# |                           End of File                                |
# +======================================================================+