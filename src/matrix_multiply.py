# [DESC]

# Import packages
import random
import sys
from timeit import default_timer as timer
from pyspark.mllib.linalg import Vectors
from scipy.sparse import lil_matrix
from pyspark.sql import SparkSession
from pyspark.mllib.linalg.distributed import *

# Define matrix dimensions
N = 500

#
def create_matrix(size):
    return [[random.randint((100 * -1), 100) for i in range(size)] \
            for j in range(size)]

#
def create_empty_matrix(size):
    return [[0 for i in range(size)] for j in range(size)]

#
A = create_matrix(N)
B = create_matrix(N)
C = create_empty_matrix(N)

#
def matrix_multiply(A, B, C, size):
    for i in range(size):
        for j in range(size):
            for k in range(size):
                C[i][j] += A[i][k] * B[k][j]

    return C

print('Performing standard matrix multiplication')

#
start = timer() #
C = matrix_multiply(A, B, C, N) #
end = timer() #

#
print('Wall time:', end - start, '\nResult:')
#print("C:")
#for i in range(N):
#    print(C[i])

#
app_name = 'PySpark Matrix Multiplication Example'
master = 'local'

# Taken from Stack Overflow:
# How to Multiply Two Numpy Matrices in PySpark?
# https://stackoverflow.com/questions/42889965/multiply-two-numpy-matrices-in-pyspark
def as_block_matrix(rdd, rows = N, columns = N):
    return IndexedRowMatrix(
        rdd.zipWithIndex().map(lambda i: IndexedRow(i[1], i[0]))
    ).toBlockMatrix(rows, columns)

# Taken from Stack Overflow:
# How to Convert a DistributedMatrix to Scipy Sparse or Numpy Array
# https://stackoverflow.com/questions/54083978/convert-a-distributedmatrix-to-scipy-sparse-or-numpy-array
def indexedrowmatrix_to_array(matrix):
    output = lil_matrix((matrix.numRows(), matrix.numCols()))
    for indexed_row in matrix.rows.collect():
        output[indexed_row.index] = indexed_row.vector
    return output

# Create Spark session
spark = SparkSession.builder.appName(app_name).master(master).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#
A_rdd = spark.sparkContext.parallelize(A)
B_rdd = spark.sparkContext.parallelize(B)

#
start = timer() #
C_matrix = as_block_matrix(A_rdd).multiply(as_block_matrix(B_rdd)) #
end = timer() #

#
print('Wall time:', end - start, '\nResult:')
#print("C_matrix:")

#
result = indexedrowmatrix_to_array(C_matrix.toIndexedRowMatrix())

#
#print(result)
print('E.O.O.')