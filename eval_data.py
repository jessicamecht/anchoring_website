import numpy as np 
import pandas as pd 
import sklearn
import os
from rl_anchoring.action_selection import predict
from rl_anchoring.plot import * 


def load_data(path):
    files = []
    for folder in os.listdir(path):
        for filename in os.listdir(path + folder):
            if filename.endswith(".csv"): 
                df = pd.read_csv(os.path.join(path,folder,filename))
                if len(df.columns) == 6:
                    df.columns = ["idx", "itemNumber", "reviewText", "summary", "originalRating", "turker_decision"]
                elif len(df.columns) == 4:
                    df.columns = ["idx", "summary", "reviewText", "turker_decision"]
                    data_paths = ['books_reviews.csv', 'books_reviews_2.csv', 'books_reviews_3.csv', 'books_reviews_4.csv', 'books_reviews_5.csv', 'books_reviews_6.csv', 'books_reviews_7.csv', 'books_reviews_8.csv',
                    'books_reviews_9.csv', 'books_reviews_10.csv']
                    for path_orig in data_paths:
                        if path_orig in filename:
                            rating = pd.read_csv(f"./review_sessions/{path_orig}")['overall']
                            df["originalRating"] = rating
                    df = df[["idx", "reviewText", "summary",  "originalRating", "turker_decision"]]

                predictions, dec_f = predict(np.array(df))
                df["prediction"] = predictions
                df["svm_confidence"] = dec_f
                #svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score
                df = df[["prediction", "svm_confidence", "reviewText", "originalRating", "summary", "idx", "turker_decision"]]
                files.append(df)
                continue
            else:
                continue
    return files

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



if __name__ == "__main__":
    path = "./data/"
    data = load_data(path)
    
    #generate_plot(data, "heuristic_resample")
    #print("Heuristically resampled agreement: ", agreement(data))
    #print("Heuristically resampled agreement with svm: ", agreement_svm(data))

    ai_path ="./data/data_ai/"
    data_ai = load_data(ai_path)
    #generate_plot(data_ai, "rl_resample")
    #print("RL resampled agreement: ", agreement(data_ai))
    #print("RL resampled agreement with svm: ", agreement_svm(data_ai))

    plot_agreement(data, "agreement_heuristic")
    plot_agreement(data_ai, "agreement_rl")

    




    