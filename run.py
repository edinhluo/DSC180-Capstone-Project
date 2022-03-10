import sys
import json
import os

sys.path.insert(0, 'src')
from nonadaptive import run_nonadaptive
from adaptive import run_adaptive
from disjoint import run_equal
import simple_pool_func as sp
import degree_centrality_pool as dc
import network as net
import numpy as np

from calculate_tests import tensor_simulation_3d
from predict_model_tensor import predict_tensor_output
from train_model_tensor import tensor_simulation_2d

import extendedNet_Pipe as ep
import BRC_pipe as bp

def main(targets):
    
    nonadaptive_config = json.load(open('config/nonadaptive-params.json'))
    adaptive_config = json.load(open('config/adaptive-params.json'))
    equal_config = json.load(open('config/equal-params.json'))
    network_config = json.load(open('config/network_params.json'))
    degree_config = json.load(open('config/degree_params.json'))
    extended_config = json.load(open('config/extendedNet_params.json'))

    
    if 'nonadaptive' in targets:
        run_nonadaptive(**nonadaptive_config)
    if 'adaptive' in targets:
        run_adaptive(**adaptive_config)
    if 'recursive' in targets:
        run_equal(**equal_config)

    if 'network' in targets:
        net.disease_spread(**network_config)
    if 'degree' in targets:
        dc.test(**degree_config)

    if 'extendedNet' in targets:
        ep.extendedNet(**extended_config)

    if 'BRC' in targets:
        bp.BRFpipe(**extended_config)
        
    if 'tensor' in targets:
        if 'predict' in targets:
            print('Enter the location of the pooling matrix:')
            A_matrix = input()
            print('Enter the location of the pooled output vector')
            Y_vector = input()
            
            with open(A_matrix, 'r') as f:
                pools = f.readlines()
                for i in range(len(pools)):
                    pools[i] = list(map(int, list(pools.strip())))
                A = np.array(pools)
            
            with open(Y_vector, 'r') as f:
                output_Y = f.readlines()
                for vector in output_Y:
                    Y = np.array(list(map(int, list(vector.strip()))))
            
            print(predict_tensor_output(A, Y))
                    

        else:
            print('Enter the amount of samples:')
            sample = input()
            try:
                sample_int = int(input)
                print(tensor_simulation_2d(sample_int))
            except ValueError:
                print("Input is not an integer.")

    if 'test' in targets:
        run_nonadaptive(**nonadaptive_config)
        run_adaptive(**adaptive_config)
        run_equal(**equal_config)
        net.disease_spread(**network_config)
        dc.test(**degree_config)
        ep.extendedNet(**extended_config)
        bp.BRFpipe(**extended_config)

        with open('test/testdata/Sample961Infected4.txt', "r") as f:
            str_X = f.readline()
        X = list(map(int, str_X.strip('[]').split(', ')))
        # write to disk
        # do a function
        print(tensor_simulation_3d(X))
        
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
