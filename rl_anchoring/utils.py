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
    read_dictionary = np.load('../review_data_mturk/mturk_review_data_w_10_better_svm_scores_1.npy',allow_pickle='TRUE')
    return read_dictionary


def get_input_output_data(review_session, input_for_lstm):
    ### SVM Score for Student ###################################
    svm_decision = np.array(review_session)[1:,-2]
    svm_decision = torch.Tensor(np.array(svm_decision, dtype=int)).unsqueeze(1)
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

def get_input_output_data_items_w_features(review_session):
    '''svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score'''
    ### SVM Score ###################################
    features = review_session[1:,2]
    svm_decision = review_session[1:,0]
    svm_decision = torch.tensor(np.array(svm_decision, dtype=float)).to(device).to(torch.float).unsqueeze(1)
    ### Reviewer Decisions ####################################
    reviewer_decision = torch.tensor(np.array(review_session[1:,-1], dtype=float)).to(device).to(torch.long) 

    ### Previous Decisions ####################################
    previous_decisions = torch.tensor(np.array(review_session[:-1,-1], dtype=float)).to(device).to(torch.float).unsqueeze(1)
    lstm_input = torch.cat((previous_decisions, svm_decision), 1)
    return lstm_input, reviewer_decision, features


def correlation(review_sessions):
    '''mearures the correlation between the number of decisions since last accept 
    and the actual decision'''
    #timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
    decision = []
    lstm_decisions = []
    number_steps_since_admission = []
    number_steps_since_admission_svm = []
    number_steps_since_admission_lstm = []
    svm_decisions = []
    svm_confidences = []
    for session in review_sessions:
        decision.extend(np.array(session)[:,1] > 1)
        if np.array(session).shape[1] == 7:
            svm_decisions.extend(np.array(session)[:,-3])
            svm_confidences.extend(np.array(session)[:,-2])
            lstm_decisions.extend(np.array(session)[:,-1])
        else:
            svm_decisions.extend(np.array(session)[:,-2])       
            svm_confidences.extend(np.array(session)[:,-1])
        n_steps_human = 0
        n_steps_svm = 0
        n_steps_lstm = 0
        for student in session:
            if len(student) == 6:
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence = student
            else:
                timestamp, target_decision, final_decision, features, svm_decision, svm_confidence, lstm_decision = student
                if lstm_decision == 1:
                    n_steps_lstm = 0
                else:
                    n_steps_lstm+=1
            if target_decision > 1:
                n_steps_human = 0
            else:
                n_steps_human+=1
            if svm_decision == 1:
                n_steps_svm = 0
            else:
                n_steps_svm+=1
            number_steps_since_admission.append(n_steps_human)
            number_steps_since_admission_svm.append(n_steps_svm)
            number_steps_since_admission_lstm.append(n_steps_lstm)
    assert(len(decision) == len(number_steps_since_admission))
    r = np.corrcoef(decision, number_steps_since_admission)[0][1]
    print(f"Correlation between human decisions and the number of steps since last human admission", r)
    print(np.array(svm_decisions).shape, np.array(number_steps_since_admission_svm).shape)
    r1 = np.corrcoef(svm_decisions, number_steps_since_admission_svm)[0][1]
    print("Correlation between svm decisions and the number of steps since last svm admission", r1)
    r2 = np.corrcoef(svm_confidences, number_steps_since_admission)[0][1]
    print(f"Correlation between svm confidence and the number of steps since last human admission", r2)
    r3 = np.corrcoef(svm_confidences, number_steps_since_admission_lstm)[0][1]
    print(f"Correlation between svm confidence and the number of steps since last lstm admission", r3)
    return r, r1, r2, r3

