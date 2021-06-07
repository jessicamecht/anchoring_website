import numpy as np 
import pandas as pd 
import sklearn
import os
from rl_anchoring.action_selection import predict
from rl_anchoring.utils import *
from eval_utils import * 

def get_agreement(data, filename, key="turker_decision"):
    agreement_per_class_good = 0
    agreement_per_class_mid = 0
    agreement_per_class_bad = 0

    den_good = 0
    den_bad = 0
    den_mid = 0

    for review in data:
        decisions = review["originalRating"] > 3
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

    ai_path ="./data/data_ai/"
    data_ai = load_data(ai_path)
    ba, ma, ga = get_agreement(data_ai, "agreement_rl")

    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')
    reviews = []
    for review in data: 
        df = pd.DataFrame(review)
        df.columns =['prediction', 'svm_confidence', 'features', 'target_decision', 'originalRating', 'Item_Number', 'turker_decision']
        reviews.append(df)
    ba_1, ma_1, ga_1 = get_agreement(reviews, "agreement_original")
    ba_3, ma_3, ga_3 = get_agreement(reviews, "agreement_original", 'prediction')

    x_labels = ["Clear Reject", "Borderline", "Clear Admit"]
    y = [ba, ma, ga]

    x = [0,1,2]
    plt.xticks(x,x_labels)
    plt.xlabel("Original Rating")
    plt.ylabel("Agreement with Original Review")
    plt.plot(x,y, linestyle='--', marker='o', color="black")

    y_1 = [ba_1, ma_1, ga_1]
    plt.plot(x,y_1, linestyle='--', marker='o', color='grey')

    y_2 = [0.9119420989143546, 0.8492307692307692, 0.579155672823219]
    plt.plot(x,y_2, linestyle='--', marker='o', color="tab:blue")

    y_3 = [ba_3, ma_3, ga_3]
    plt.plot(x,y_3, linestyle='--', marker='o', color="royalblue")

    plt.legend(["LSTM+RL Resampled","Original", "Debiased", "SVM"])
    plt.savefig("agreement_to_orig.png")


    plt.close()


    




    