import numpy as np
import pandas as pd
import tensorflow as tf
import math


def tensor_simulation_2d(sample_size, split=False):
    if split == True:
        print('Splitting samples into two groups. Preparing A matrix.')
        size_1 = sample_size//2
        size_2 = sample_size - (sample_size//2)
        
        A_1 = tensor_simulation_2d(size_1, split=False)
        A_1_appended = np.array([list(pool) + [0]*size_2 for pool in A_1])
        
        A_2 = tensor_simulation_2d(size_2, split=False)
        A_2_appended = np.array([[0]*size_1 + list(pool) for pool in A_2])
        
        A = np.append(A_1_appended, A_2_appended, axis=0)
        
    
    else:
        samples = iter(range(sample_size))

        matrix_dimension = math.ceil(sample_size**(1/2))
        total_tests = matrix_dimension*2
        print("Creating 2d tensor classification with " + str(sample_size) + " samples, " + str(matrix_dimension) + " initial pool size, and "+ str(matrix_dimension * 2) + " initial pools.")

        matrix = []
        for i in range(matrix_dimension):
            row = []
            for j in range(matrix_dimension):
                row.append(next(samples, np.random.choice(range(sample_size))))
            matrix.append(row)

        A = np.array([[0 for i in range(sample_size)] for j in range(matrix_dimension * 2)])
        tests = iter(range(matrix_dimension*2))

        for i in range(matrix_dimension):
            lst = tf.constant(matrix)[i, :].numpy()
            test = next(tests)

            for sample in lst:
                if sample >= sample_size:
                    break
                A[test][sample] = 1

        for i in range(matrix_dimension):
            lst = tf.constant(matrix)[:, i].numpy()
            test = next(tests)

            for sample in lst:
                if sample >= sample_size:
                    break
                A[test][sample] = 1
    
    with open('A_matrix.txt', 'w') as f:
        for pool in A:
            for sample in pool:
                f.write(str(sample))
            f.write('\n')

    return A