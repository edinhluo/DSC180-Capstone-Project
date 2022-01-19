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

from calculate_tests import tensor_simulation_3d

def main(targets):
    
    nonadaptive_config = json.load(open('config/nonadaptive-params.json'))
    adaptive_config = json.load(open('config/adaptive-params.json'))
    equal_config = json.load(open('config/equal-params.json'))
    network_config = json.load(open('config/network_params.json'))
    degree_config = json.load(open('config/degree_params.json'))

    
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

    if 'test' in targets:
        run_nonadaptive(**nonadaptive_config)
        run_adaptive(**adaptive_config)
        run_equal(**equal_config)
        net.disease_spread(**network_config)
        dc.test(**degree_config)

        with open('test/testdata/Sample961Infected4.txt', "r") as f:
            str_X = f.readline()
        X = list(map(int, str_X.strip('[]').split(', ')))
        # write to disk
        # do a function
        print(tensor_simulation_3d(X))
        
if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
