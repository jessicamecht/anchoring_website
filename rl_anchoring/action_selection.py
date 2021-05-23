import torch 
import numpy as np
from torch.distributions import Categorical
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(output):
    return torch.argmax(output)
    #return torch.tensor(output.max(2)[1].view(1), dtype=torch.int64, device=device)


def get_next_action(lstm_input, pool, hidden_anchor_state, possible_next_instances_mask, actor, anchor_lstm):
    actor.eval()
    
    anchor_lstm.eval()
    predictions, dec_f = predict(pool)

    lstm_input= lstm_input.cpu().reshape(len(lstm_input), 1)
    predictions= predictions.reshape(len(predictions), 1)
    input_to_lstm = np.append(lstm_input, predictions, axis=1)
    input_to_lstm = torch.tensor(input_to_lstm, device=device)

    with torch.no_grad():
        redictions, (state, _), _ = anchor_lstm(input_to_lstm,hidden_anchor_state)
        output = actor(state[:,-1:,:])
        #eliminate impossible actions (already sampled students and students from a different year)

        #Select student to be sampled
        valid_output = output * torch.tensor(possible_next_instances_mask).to(device)
    
        #Select student to be sampled
        valid_output = valid_output[0:len(pool)]
        action_idx = select_action(valid_output).item()
    return action_idx


def predict(pool):
    pool = np.array(pool)
    print(pool)
    pkl_filename = './rl_anchoring/state_dicts/svm_all_unbalanced.pkl'
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)
    dec_f = clf.decision_function(pool[:,2])
    dec_f = dec_f.reshape((len(dec_f),1))
    predictions = clf.predict(pool[:,2])
    return predictions, dec_f

def sort_by_confidence(pool, possible_instances_mask):
    predictions, dec_f = predict(pool)
    possible_instances_mask = np.array(possible_instances_mask).astype(bool)
    
    idx = np.array(list(range(len(possible_instances_mask))))
    possible_idxs = np.array(idx)[possible_instances_mask]
    possible_idxs = possible_idxs.reshape((possible_idxs.shape[0],1))

    possible_dec_f = np.array(dec_f)[possible_instances_mask]
    possible_dec_f = possible_dec_f.reshape((possible_dec_f.shape[0],1))

    
    predictions = np.array(predictions)[possible_instances_mask]
    predictions = predictions.reshape((predictions.shape[0],1))

    possible_dec_f = np.append(possible_dec_f, possible_idxs, axis=1)
    possible_dec_f = np.append(possible_dec_f, predictions, axis=1)

    rejects_mask = possible_dec_f[:,-1] == 0 # select based on SVM score 
    admit_mask = possible_dec_f[:,-1] == 1 

    reject = possible_dec_f[rejects_mask]
    admit = possible_dec_f[admit_mask]

    sorted_pool_reject = sorted(reject, key=lambda x: x[0], reverse = True) #sort by confidence
    sorted_pool_admit = sorted(admit, key=lambda x: x[0], reverse = True)
    

    return np.array(sorted_pool_reject)[:,1] if sorted_pool_reject != [] else [], np.array(sorted_pool_admit)[:,1] if sorted_pool_admit != [] else []

def heuristic_select_next_action(last_decision, possible_instances, possible_instances_mask):
    '''selects an instance based on the last decision'''
    sorted_pool_reject, sorted_pool_admit = sort_by_confidence(possible_instances, possible_instances_mask)
    print(sorted_pool_reject, sorted_pool_admit)
    for possible in possible_instances_mask:
        if possible == 0:
            continue
        if len(sorted_pool_admit) == 0:
            idx =  sorted_pool_reject[0]
        elif len(sorted_pool_reject) == 0:
            idx =  sorted_pool_admit[0]
        elif last_decision == 1:
            idx =  sorted_pool_reject[0]
        else: 
            idx =  sorted_pool_admit[0]
    return int(idx)