import numpy as np 
import torch
import collections
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from random import randrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    '''loads data in format:
    "reviewer: [timestamp, target_grade, target_decision, [features]]
    features are GPA, SAT, ...
    target_grade is the rating which was given to the student by this reviewer
    target_decision is if the student was actually admitted  
    '''
    read_dictionary = np.load('../admissions.npy',allow_pickle='TRUE').item()
    return read_dictionary

def load_admissions():
    #timestamp, reviewer_score, final_decision, features, svm_predictions, svm_confidence
    '''loads data in format:
    "reviewer: [timestamp, target_grade, target_decision, [features]]
    features are GPA, SAT, ...
    target_grade is the rating which was given to the student by this reviewer
    target_decision is if the student was actually admitted  
    '''
    read_dictionary = np.load('../review_data_mturk/admissions_without_reviewer.npy',allow_pickle='TRUE')
    return read_dictionary

def load_data_items():
    '''loads data in format:
    "[[svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score]]
    features are review text
    final_decision is the original review
    target_decision: majority vote of turkers 
    reviewer_score is the single turker review
    '''
    read_dictionary = np.load('../review_data_mturk/mturk_review_data_w_16_unbalancedall_unbalanced.npy',allow_pickle='TRUE')
    return read_dictionary

def augment_data(data):
    '''select random crops'''
    more_sessions = []
    for review_session in data:
        if len(review_session) < 3:
            continue
        random_length = randrange(len(review_session)-1)
        more_sessions.append(review_session[:random_length])
    return np.array(more_sessions)




def get_input_output_data(review_session):

    elem = review_session
    df = pd.DataFrame(elem[:,4], columns = ["svm_decision"]) 
    df["timestamp"] = elem[:,0]
    df["features"] = elem[:,3]
    df["svm_confidence"] = elem[:,5]
    df["final_decision"] = elem[:,2]
    df["reviewer_score"] = elem[:,1]

    ### SVM Score for Student ###################################
    svm_decision = torch.Tensor(np.array(df["svm_decision"][1:], dtype=int)).unsqueeze(1).to(device)
    ### Reviewer Decisions for Students ####################################
    reviewer_decision = torch.Tensor(np.array(df["reviewer_score"])[1:] > 1).to(device).to(torch.int64) 
    ### Previous Decisions Score for Student ####################################
    previous_decisions = torch.tensor(np.array(df["reviewer_score"][:-1], dtype=float) > 1).to(device).to(torch.float).unsqueeze(1)
    lstm_input = torch.cat((previous_decisions, svm_decision), 1)
    conf = torch.tensor(np.array(df["svm_confidence"][:-1], dtype=float)).to(device).unsqueeze(1).to(torch.float)

    #lstm_input = torch.cat((lstm_input, conf), 1)

    exp_dec_ft = exp_decay_features(reviewer_decision)
    #lstm_input = torch.cat((lstm_input, exp_dec_ft), 1)

    return lstm_input, reviewer_decision

def get_input_output_data_items(review_session):
    '''svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score'''
    ### SVM Score ###################################
    elem = review_session
    df = pd.DataFrame(elem[:,0], columns = ["svm_decision"]) 
    df["svm_confidence"] = elem[:,1]
    df["features"] = elem[:,2]
    df["item_number"] = elem[:,3]
    df["original_review"] = elem[:,4]
    df["item_number"] = elem[:,5]
    df["turker_review"] = elem[:,6]

    svm_decision = torch.tensor(np.array(df["svm_decision"][1:], dtype=float)).to(device).to(torch.float).unsqueeze(1)
    ### Reviewer Decisions ####################################
    reviewer_decision = torch.tensor(np.array(df["turker_review"][1:], dtype=int)).to(device)

    ### Previous Decisions ####################################
    previous_decisions = torch.tensor(np.array(df["turker_review"][:-1], dtype=int)).to(device).unsqueeze(1)
    conf = torch.tensor(np.array(df["svm_confidence"][:-1], dtype=float)).to(device).unsqueeze(1).to(torch.float)


    lstm_input = torch.cat((previous_decisions, svm_decision), 1)
    #lstm_input = torch.cat((lstm_input, conf), 1)

    exp_dec_ft = exp_decay_features(reviewer_decision)
    #lstm_input = torch.cat((lstm_input, exp_dec_ft), 1)

    return lstm_input, reviewer_decision

def exp_decay_fn(x, b):
    '''exponential decay function to determine the anchor '''
    return b*np.exp(-b*x)

def exp_decay_features(reviewer_decision):
    n_steps = 0
    n_steps_since_pos = []
    for dec in reviewer_decision:
        n_steps_since_pos.append(exp_decay_fn(n_steps, 0.55))
        if dec:
            n_steps = 0 
        else:
            n_steps+=1
    return torch.tensor(n_steps_since_pos).to(device).unsqueeze(1).to(torch.float)
