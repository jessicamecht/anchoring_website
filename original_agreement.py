import numpy as np
from rl_anchoring.plot import * 
import pandas as pd

def agreement_svm(data):
    all_agreement = 0
    denominator = 0
    for review in data:
        #svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score

        all_agreement += ((review[:,0]) == (review[:,-1])).sum()
        denominator += review[:,0].shape[0]
    return all_agreement/denominator

if __name__ == "__main__":
    data = np.load('./review_data_mturk/mturk_review_data_w_10_all_unbalanced.npy',allow_pickle='TRUE')
    reviews = []
    for review in data: 
        df = pd.DataFrame(review)
        df.columns =['prediction', 'svm_confidence', 'features', 'target_decision', 'originalRating', 'Item_Number', 'turker_decision']
        reviews.append(df)
    print("Original agreement: ", agreement_svm(data))
    plot_agreement(reviews, "agreement_original")
