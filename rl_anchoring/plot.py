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
    keyNames = ['0','1','2','3','4','5-15', "> 15"]
    #keyNames = ['0','1','2','3','4','5','6','7','8', '9', '10', '11','12','13','14','15','16', "> 16"]


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


def plot_n_decisions_vs_confidence(review_sessions, figname='./figures/resampled_confidence.png', lstm=False):
    scoresByInterval = collections.defaultdict(list)
    for session in review_sessions:
        nSinceAccept = None
        for i in range(len(session)):
            timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = session[i]
            accept = target_decision > 1 if not lstm else target_decision
            svmScore = svm_confidence
            if accept:
                if nSinceAccept == None:
                    # Skip the first one
                    nSinceAccept = 0
                    continue
                binSinceAccept = str(nSinceAccept)
                if nSinceAccept > 5 and nSinceAccept <= 10:
                    binSinceAccept = "6-10"
                elif nSinceAccept > 10 and nSinceAccept <= 20:
                    binSinceAccept = "11-20"
                elif nSinceAccept > 20:
                    binSinceAccept = "> 20"
                scoresByInterval[binSinceAccept].append(svmScore)
                nSinceAccept = 0
            else:
                if nSinceAccept != None:
                    nSinceAccept += 1
                    binSinceAccept = str(nSinceAccept)
                    if nSinceAccept > 5 and nSinceAccept <= 10:
                        binSinceAccept = "6-10"
                    if nSinceAccept > 10 and nSinceAccept <= 20:
                        binSinceAccept = "11-20"
                    elif nSinceAccept > 20:
                        binSinceAccept = "> 20"
                    scoresByInterval[binSinceAccept].append(svm_confidence)
    averageScoresByInterval = {}

    for inter in scoresByInterval:
        averageScoresByInterval[inter] = sum(scoresByInterval[inter]) / len(scoresByInterval[inter])
    
    keyNames = ["0", "1", "2", "3", "4", "5", "6-10", "11-20", "> 20"]
    keyNames = [name for name in keyNames if name in list(averageScoresByInterval.keys())]
    ks = list(range(len(keyNames)))
    values = [averageScoresByInterval[kn] for kn in keyNames]
    plt.xticks(ks,keyNames)
    plt.xlabel("Numbers of decisions since last accept")
    plt.ylabel("average SVM confidence of accepted file")
    plt.bar(ks, values)
    plt.savefig(figname)
    plt.close()
