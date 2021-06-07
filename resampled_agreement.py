import numpy as np 
import pandas as pd 
import sklearn
import os
from rl_anchoring.action_selection import predict
from rl_anchoring.plot import * 
from rl_anchoring.utils import *

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
    ba, ma, ga = plot_agreement(data_ai, "agreement_rl")

    data = np.load('./review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')
    reviews = []
    for review in data: 
        df = pd.DataFrame(review)
        df.columns =['prediction', 'svm_confidence', 'features', 'target_decision', 'originalRating', 'Item_Number', 'turker_decision']
        reviews.append(df)
    ba_1, ma_1, ga_1 = plot_agreement(reviews, "agreement_original")
    ba_3, ma_3, ga_3 = plot_agreement(reviews, "agreement_original", 'prediction')

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


    




    