import torch 
import numpy as np
from torch.distributions import Categorical

def sort_by_confidence(pool, possible_instances_mask, svm_pred_idx = 0, svm_confidence_idx = 1):
    enumerated_pool = np.array(list(enumerate(pool)))[possible_instances_mask]
    rejects_mask = enumerated_pool[:,1][:,svm_pred_idx] == 0 # select based on SVM score 
    admit_mask = enumerated_pool[:,1][:,svm_pred_idx] == 1 

    reject = enumerated_pool[rejects_mask]
    admit = enumerated_pool[admit_mask]
    sorted_pool_reject = sorted(reject,  key = lambda x : x[1][svm_confidence_idx])#sort by confidence
    sorted_pool_admit = sorted(admit,  key = lambda x : x[1][svm_confidence_idx])
    return sorted_pool_reject, sorted_pool_admit

def heuristic_select_next_action(last_decision, possible_instances, possible_instances_mask):
    '''selects an instance based on the last decision'''
    sorted_pool_reject, sorted_pool_admit = sort_by_confidence(possible_instances, possible_instances_mask)
    for idx, possible in enumerate(possible_instances_mask):
        if possible == 0:
            continue
        if len(sorted_pool_admit) == 0:
            idx, instance =  sorted_pool_reject[idx]
        elif len(sorted_pool_reject) == 0:
            idx, instance =  sorted_pool_admit[idx]
        elif last_decision == 1:
            idx, instance =  sorted_pool_reject[idx]
        else: 
            idx, instance =  sorted_pool_admit[idx]
    return idx