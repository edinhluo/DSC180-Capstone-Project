import numpy as np
import pandas as pd
import tensorflow as tf
import math

def tensor_simulation_3d(X):
    
    sample_size = len(X)
    infected = sum(X)
    samples = iter(range(sample_size))
    
    tensor_dimension = math.ceil(sample_size**(1/3))
    total_tests = tensor_dimension*3

    f = open("test/testdata/test_output.txt", "w")
    sim_3d_output = "simulating 3d tensor classification with " + str(sample_size) + " samples and " + str(tensor_dimension**2) + " initial pool size."
    f.write(sim_3d_output + '\n')
    print(sim_3d_output)


    tensor = []
    for k in range(tensor_dimension):
        matrix = []
        for i in range(tensor_dimension):
            row = []
            for j in range(tensor_dimension):
                row.append(next(samples, np.random.choice(range(sample_size))))
            matrix.append(row)
        tensor.append(matrix)
        
    def flatten(tensor_2d):
        lst = []
        for i in tensor_2d:
            for j in i:
                lst.append(j)
        return lst
    
    A = np.array([[0 for i in range(sample_size)] for j in range(tensor_dimension * 3)])
    tests = iter(range(tensor_dimension*3))

    for i in range(tensor_dimension):
        lst = flatten(tf.constant(tensor)[i, :, :].numpy())
        test = next(tests)

        for sample in lst:
            if sample >= sample_size:
                break
            A[test][sample] = 1

    for i in range(tensor_dimension):
        lst = flatten(tf.constant(tensor)[:, i, :].numpy())
        test = next(tests)

        for sample in lst:
            if sample >= sample_size:
                break
            A[test][sample] = 1

    for i in range(tensor_dimension):
        lst = flatten(tf.constant(tensor)[:, :, i].numpy())
        test = next(tests)

        for sample in lst:
            if sample >= sample_size:
                break
            A[test][sample] = 1
            
    infected_samples = np.random.randint(sample_size, size=infected)
    X = np.array([1 if i in infected_samples else 0 for i in range(sample_size)])
    
    Y = ((np.matmul(A, X.T) > 0)*1)
    
    not_infected = set()
    for test_num in range(len(Y)):
        if Y[test_num] == 0:
            not_infected.update([i for i in range(len(A[test_num])) if A[test_num][i] == 1])
            
    end_3d_output = "3d tensor ended with " + str(len(not_infected)) + " labeled as not infected"
    f.write(end_3d_output + '\n')
    print(end_3d_output)
    
    maybe_infected = [i for i in range(len(X)) if i not in not_infected]
    step_2 = math.ceil(len(maybe_infected)**(1/2))

    sim_2d_output = "simulating 2d tensor classification with " + str(len(maybe_infected)) + " samples and " + str(step_2) + " initial pool size."
    f.write(sim_3d_output + '\n')
    print(sim_2d_output)


    enumerated_ties = list(enumerate(maybe_infected))
    samples = iter(range(len(enumerated_ties)))

    matrix = []
    for i in range(step_2):
        row = []
        for j in range(step_2):
            row.append(next(samples, np.random.choice(maybe_infected)))
        matrix.append(row)

    A_2 = np.array([[0 for i in range(len(maybe_infected))] for i in range(step_2*2)])
    
    total_tests += step_2*2
    tests = iter(range(step_2*2))

    for i in range(step_2):
        lst = tf.constant(matrix)[i, :].numpy()
        test = next(tests)

        for sample in lst:
            if sample >= len(maybe_infected):
                break
            A_2[test][sample] = 1

    for i in range(step_2):
        lst = tf.constant(matrix)[:, i].numpy()
        test = next(tests)

        for sample in lst:
            if sample >= len(maybe_infected):
                break
            A_2[test][sample] = 1
            
    X_2 = np.array([1 if maybe_infected[i] in infected_samples else 0 for i in range(len(maybe_infected))])
    Y_2 = ((np.matmul(A_2, X_2.T) > 0)*1)

    not_infected_2 = set()
    for test_num in range(len(Y_2)):
        if Y_2[test_num] == 0:
            not_infected_2.update([i for i in range(len(A_2[test_num])) if A_2[test_num][i] == 1])

    end_2d_output = "2d tensor ended with " + str(len(not_infected_2)) + " labeled as not infected"
    f.write(end_2d_output + '\n')
    print(end_2d_output)
    
    maybe_infected_2 = [i for i in range(len(X_2)) if i not in not_infected_2]
    total_tests += len(maybe_infected_2)
    
    write_tests = 'total number of tests required: ' + str(total_tests)
    f.write(write_tests)
    f.close()

    return write_tests