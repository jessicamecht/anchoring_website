import numpy as np 
import torch
import collections
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import pandas as pd
from rl_anchoring.action_selection import predict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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