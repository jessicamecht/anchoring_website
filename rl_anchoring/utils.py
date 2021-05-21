import numpy as np 
import torch
import collections
import matplotlib.pyplot as plt
from collections import defaultdict

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
    read_dictionary = np.load('../review_data_mturk/mturk_review_data_w_10_all_unbalanced.npy',allow_pickle='TRUE')
    return read_dictionary


def get_input_output_data(review_session, input_for_lstm):
    ### SVM Score for Student ###################################
    svm_decision = np.array(review_session)[1:,-2]
    svm_decision = torch.Tensor(np.array(svm_decision, dtype=int)).unsqueeze(1).to(device)
    ### Reviewer Decisions for Students ####################################
    reviewer_decision = torch.Tensor(np.array(review_session)[1:,1] > 1).to(device).to(torch.int64) 
    ### Previous Decisions Score for Student ####################################
    previous_decisions = torch.tensor(np.array(review_session[:-1,-1], dtype=float) > 1).to(device).to(torch.float).unsqueeze(1)
    lstm_input = torch.cat((previous_decisions, svm_decision), 1)
    return lstm_input, reviewer_decision

def get_input_output_data_items(review_session):
    '''svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score'''
    ### SVM Score ###################################
    svm_decision = review_session[1:,0]
    svm_decision = torch.tensor(np.array(svm_decision, dtype=float)).to(device).to(torch.float).unsqueeze(1)
    ### Reviewer Decisions ####################################
    reviewer_decision = torch.tensor(np.array(review_session[1:,-1], dtype=float)).to(device).to(torch.long) 

    ### Previous Decisions ####################################
    previous_decisions = torch.tensor(np.array(review_session[:-1,-1], dtype=float)).to(device).to(torch.float).unsqueeze(1)
    lstm_input = torch.cat((previous_decisions, svm_decision), 1)
    return lstm_input, reviewer_decision

