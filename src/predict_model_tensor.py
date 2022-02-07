def predict_tensor_output(A, Y):
    not_infected = set()
    for test_num in range(len(Y)):
        if Y[test_num] == 0:
            not_infected.update([i for i in range(len(A[test_num])) if A[test_num][i] == 1])

    maybe_infected = [i for i in range(len(X)) if i not in not_infected]
    print('Please individually test the following samples')

    return np.array([1 if i in maybe_infected else 0 for i in range(len(A[0]))])