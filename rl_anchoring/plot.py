import numpy as np 
import torch
import collections
import matplotlib.pyplot as plt
from collections import defaultdict


def generate_plot(all_review_sessions, filename):
    scoresByInterval = defaultdict(list)
    all_s = []
    for j, review_session in enumerate(all_review_sessions):
        nSinceAccept = None
        for i in range(len(review_session)):
            review_session = np.array(review_session)
            if len(review_session[i]) == 7:
                svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score  = review_session[i]
            if len(review_session[i]) == 8:# if we are trying to evaluate the simulation 
                svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score_human, reviewer_score  = review_session[i]
            accept = reviewer_score
            if accept:
                if nSinceAccept == None:
                    # Skip the first one
                    nSinceAccept = 0
                    continue
                binSinceAccept = str(nSinceAccept)
                if nSinceAccept >= 3 and nSinceAccept <= 4:
                    binSinceAccept = "3-4"
                if nSinceAccept >= 5 and nSinceAccept <= 15:
                    binSinceAccept = "5-15"
                if nSinceAccept > 15:
                    binSinceAccept = "> 15"
                scoresByInterval[binSinceAccept].append(svm_confidence)
                nSinceAccept = 0
            else:
                if nSinceAccept != None: 
                    nSinceAccept += 1

    numberScoresByInterval = {}

    for inter in scoresByInterval:
        numberScoresByInterval[inter] =len(scoresByInterval[inter])
    print(numberScoresByInterval)

    averageScoresByInterval = {}

    for inter in scoresByInterval:
        averageScoresByInterval[inter] = sum(scoresByInterval[inter]) / len(scoresByInterval[inter])
    
    print(averageScoresByInterval)
    keyNames = ['0','1','2','3-4','5-15', "> 15"]

    values = []
    
    for kn in keyNames:
        if kn in averageScoresByInterval.keys():
            values.append(averageScoresByInterval[kn])
    ks = list(range(len(values)))
    keyNames = keyNames[:len(values)]

    plt.xticks(ks,keyNames)
    plt.xlabel("Numbers of decisions since last accept")
    plt.ylabel("average SVM confidence of accepted file")
    plt.bar(ks, values)
    plt.savefig(f'./figures/{filename}_new.png')
    plt.close()

def plot_agreement(data, filename):
    agreement_per_class_good = 0
    agreement_per_class_mid = 0
    agreement_per_class_bad = 0

    den_good = 0
    den_bad = 0
    den_mid = 0

    for review in data:
        decisions = review["prediction"]
        turker_decision = review["turker_decision"]
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
    x_labels = ["Clear Reject", "Borderline", "Clear Admit"]
    y = [ba, ma, ga]

    print(y)
    x = [0,1,2]
    plt.xticks(x,x_labels)
    plt.xlabel("Original Rating")
    plt.ylabel("Agreement with SVM")
    plt.plot(x,y, linestyle='--', marker='o', color='b')
    plt.savefig(f'./{filename}_new.png')
    plt.close()

