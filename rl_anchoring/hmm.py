from hmmlearn.hmm import GaussianHMM
import numpy as np
from utils import * 


def main():
    ### Init Data ###################################
    data = load_data()
    timestep, _, _, data_instance, _, _ = data["reviewer_0"][0][-1]
    keys = np.array(list(data.keys()))
    folds = np.array_split(keys, 10) #10-fold cross validation 

    accuracy_train = []
    accuracy_val = []
    for i in range(len(folds)):
        print("Fold: ", i)
        ### Train and Valid Keys ###################################
        train_keys_1 = [item for sublist in folds[0:i] for item in sublist] if i > 0 else []  
        train_keys_2 = [item for sublist in folds[i+1:] for item in sublist] if len(folds) > i+1 else [] 
        train_keys = train_keys_2 + train_keys_1
        valid_keys = folds[i] 
        train_hmm(data, train_keys)
        break

def train_hmm(data, train_keys):
    '''main training function 
    iterates over all reviewers and each review session 
    for each student of the review session, 
    '''
    all_sessions = []
    
    for reviewer in data:
        if reviewer not in train_keys:
            continue

        for review_session in data[reviewer]:
            all_sessions.append(review_session)
    model = GaussianHMM(1, "full")
    print(np.array(all_sessions).shape)
    model.fit(all_sessions[0:2]) 
    #hidden_states = model.predict(all_sessions[0])

if __name__ == "__main__":
    main()