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
    plt.savefig(f'./figures/{filename}.png')
    plt.close()
