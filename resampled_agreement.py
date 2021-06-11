import numpy as np 
import pandas as pd 
import sklearn
import os
from rl_anchoring.action_selection import predict
from rl_anchoring.utils import *
from eval_utils import * 

def get_agreement(data, key="turker_decision", to="originalRating"):
    agreement_per_class_good = 0
    agreement_per_class_mid = 0
    agreement_per_class_bad = 0

    den_good = 0
    den_bad = 0
    den_mid = 0

    for review in data:
        decisions = review[to]  if to!="originalRating" else  review[to]> 3
        turker_decision = review[key]
        good_mask = review["originalRating"] > 3
        agreement_per_class_good += (decisions[good_mask] == turker_decision[good_mask]).sum()
        den_good += len(review[good_mask])

        bad_mask = review["originalRating"] < 3
        agreement_per_class_bad += ((decisions[bad_mask]) == turker_decision[bad_mask]).sum()
        den_bad += len(review[bad_mask])

        mid_mask = review["originalRating"] == 3
        agreement_per_class_mid += ((decisions[mid_mask]) == turker_decision[mid_mask]).sum()
        den_mid += len(review[mid_mask])
    print(den_good, den_bad, den_mid)
    print(agreement_per_class_good, agreement_per_class_bad, agreement_per_class_mid)

    ga, ma, ba = agreement_per_class_good/den_good, agreement_per_class_mid/den_mid, agreement_per_class_bad/den_bad
    return (ba, ma, ga)

def agreement(data):
    all_agreement = 0
    denominator = 0
    for review in data:
        all_agreement += ((review["originalRating"] > 3) == (review["turker_decision"])).sum()
        denominator += review["originalRating"].shape[0]
    return all_agreement/denominator

def agreement_svm(data):
    all_agreement = 0
    denominator = 0
    for review in data:
        all_agreement += ((review["prediction"]) == (review["turker_decision"])).sum()
        denominator += review["prediction"].shape[0]
    return all_agreement/denominator

def agreement_svm_orig(data):
    all_agreement = 0
    denominator = 0
    for review in data:
        #svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score
        all_agreement += ((review[:,0]) == (review[:,-1])).sum()
        denominator += review[:,0].shape[0]
    return all_agreement/denominator

    


if __name__ == "__main__":
    path = "./data/"
    data = load_data(path)
    to = 'originalRating'

    #AC Agreement
    ai_path ="./data/data_ai/"
    data_ai = load_data(ai_path)
    ba, ma, ga = get_agreement(data_ai, to=to)

    #DQN Agreement
    dqn_path ="./data/data_dqn/"
    data_dqn = load_data(dqn_path)
    ba_0, ma_0, ga_0 = get_agreement(data_dqn,  to=to)

    
    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')
    reviews = []
    
    for review in data: 
        df = pd.DataFrame(review)
        df.columns =['prediction', 'svm_confidence', 'features', 'target_decision', 'originalRating', 'Item_Number', 'turker_decision']
        reviews.append(df)
    
    #Original Agreement
    ba_1, ma_1, ga_1 = get_agreement(reviews, to=to)
    #SVM Agreement
    ba_3, ma_3, ga_3 = get_agreement(reviews, key="prediction", to=to)
    
    x_labels = ["Clear Reject", "Borderline", "Clear Admit"]
    x = np.array([0,1,2])
    plt.xticks(x,x_labels)

    labels = "SVM" if to == 'prediction' else "Original Rating"
    plt.xlabel(f"Original Rating")
    plt.ylabel(f"Agreement with {labels}")

    probabilistic_to_svm = [0.78367511057499, 0.6615384615384615, 0.5910290237467019]
    probabilistic_to_orig = [0.9075190993164455, 0.8461538461538461, 0.5778364116094987]

    
    offsets = [0.25, 0.125, 0, 0.125, 0.25] if to == "originalRating" else [0,0.1875, 0.0625, 0.0625, 0.1875]
    if to == "originalRating":
        y_3 = [ba_3, ma_3, ga_3]
        print(y_3)
        plt.bar(x-offsets[0],y_3, color="lightgrey", width=0.125, align='center')
    
    y_2 = probabilistic_to_svm if to=="prediction" else probabilistic_to_orig
    plt.bar(x-offsets[1],y_2,  color="tab:blue", width=0.125, align='center')

    y_0 = [ba_0, ma_0, ga_0]
    print(y_0)
    plt.bar(x-offsets[2],y_0, color="lightskyblue", width=0.125, align='center')

    y = [ba, ma, ga]
    print(y)
    plt.bar(x+offsets[3],y, color="black", width=0.125, align='center')

    y_1 = [ba_1, ma_1, ga_1]
    print(y_1)
    plt.bar(x+offsets[4],y_1,  color='grey', width=0.125, align='center')
    plt.ylim([0.3, 1.18])

    leg = ["Probabilistic", "LSTM+DQN Resampled", "LSTM+AC Resampled","Original"] if to=="prediction" else ["SVM", "Probabilistic", "LSTM+DQN Resampled", "LSTM+AC Resampled","Original"]
    plt.legend(leg, loc='upper left')
    plt.savefig(f"agreement_to_{to}.png")


    plt.close()


    




    