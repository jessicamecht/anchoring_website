import torch 
import random 
import torch.nn as nn 
import torch.optim as optim
from itertools import count
import math 
from models.models import * 
import csv 
from utils import * 
import numpy as np
import operator
import random 
import pandas as pd
from resample import * 
from sklearn.model_selection import KFold
from plot import * 
from sklearn.metrics import precision_recall_fscore_support
from models.loss import FocalLoss

#####CONFIG#########################################
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 200
####################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_size=2

def main():
    ### Init Data ###################################
    data = load_data_items()
    data_admissions = load_admissions()
    kf = KFold(n_splits=6)
    input_size = 2
    accuracy_val = []
    accuracy_train = []
    all_hidden_states = []
    step = 0

    class_map = np.hstack(np.array(list(map(lambda review: review[:,-1], data))))
    classes_pos = np.array(class_map).sum()/len(class_map)
    classes_neg = 1-classes_pos
    class_weights = torch.tensor([classes_pos, classes_neg], dtype=torch.float32)

    review_sessions_lstm = []
    all_predictions = []
    all_decisions = []
    for train_index, test_index in kf.split(data):
        data_train = data[train_index]
        data_test = data[test_index]
        ### Load Models ###################################
        anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)#FocalLoss() #

        ### Init Optimizer ###################################
        anchor_optimizer = optim.Adam(anchor_lstm.parameters(), lr=0.01)

        #acc_train = pre_train_anchor(data_admissions, anchor_lstm, anchor_optimizer, loss_fn)
        acc_train = train_anchor(data_train, anchor_lstm, anchor_optimizer, loss_fn)
        accuracy_val.append(acc_train)
        print('acc_train', acc_train)
        acc_val, review_sessions, hidden_states, reviewer_decision, predictions  = eval_anchor(data_test, anchor_lstm, step)
        all_predictions.extend(predictions)
        all_decisions.extend(reviewer_decision)

        print('acc_val', acc_val)
        accuracy_val.append(acc_val)
        review_sessions_lstm.extend(review_sessions)
        all_hidden_states.extend(hidden_states)

        step+=1

    print(precision_recall_fscore_support(all_decisions, all_predictions))


    '''anchor_lstm = AnchorLSTM(input_size, hidden_size)
    anchor_lstm.load_state_dict(torch.load(f'./state_dicts/anchor_lstm_SVM+Decision.pt', map_location=torch.device('cpu')))
    acc_val, review_sessions, hidden_states = eval_anchor(data, anchor_lstm, 0)
    accuracy_val.append(acc_val)
    review_sessions_lstm.extend(review_sessions)    
    all_hidden_states.extend(hidden_states)'''



    '''sessions = list(map(lambda session: session[:,-1], review_sessions_lstm))
    sessions = [item for sublist in sessions for item in sublist]
    correlation_lstm_decision_and_hidden_state = np.corrcoef(sessions, all_hidden_states)[0][1]
    print('Correlation of LSTM decision and hidden state:',  correlation_lstm_decision_and_hidden_state)

    prev_dec = list(map(lambda session: np.append([0], session[0:-1, -2]), review_sessions_lstm))
    prev_dec = [item for sublist in prev_dec for item in sublist]
    correlation_previous_decisions_and_hidden_state = np.corrcoef(prev_dec, all_hidden_states)[0][1]
    print('Correlation of previous human decision and hidden state:',correlation_previous_decisions_and_hidden_state)

    prev_dec = list(map(lambda session: np.append([0], session[0:-1, -1]), review_sessions_lstm))
    prev_dec = [item for sublist in prev_dec for item in sublist]
    correlation_previous_decisions_and_hidden_state = np.corrcoef(prev_dec, all_hidden_states)[0][1]
    print('Correlation of previous LSTM decision and hidden state:',correlation_previous_decisions_and_hidden_state)

    two_prev_dec = list(map(lambda session: np.append([0,0], session[0:-2, -2]), review_sessions_lstm))
    two_prev_dec = [item for sublist in two_prev_dec for item in sublist]
    correlation_two_previous_decisions_and_hidden_state = np.corrcoef(two_prev_dec, all_hidden_states)[0][1]
    print('Correlation of two previous human decision and hidden state:',correlation_two_previous_decisions_and_hidden_state)

    two_prev_dec = list(map(lambda session: np.append([0,0], session[0:-2, -1]), review_sessions_lstm))
    two_prev_dec = [item for sublist in two_prev_dec for item in sublist]
    correlation_two_previous_decisions_and_hidden_state = np.corrcoef(two_prev_dec, all_hidden_states)[0][1]
    print('Correlation of two previous LSTM decision and hidden state:',correlation_two_previous_decisions_and_hidden_state)'''

    generate_plot(review_sessions_lstm, f"./final_confidence_items_6")

    torch.save(anchor_lstm.state_dict(), f'./state_dicts/anchor_lstm_items_6.pt')
    print("Validation Accuracy: ", np.array(accuracy_val).mean())#, "Training Accuracy: ", np.array(acc_train).mean())

def pre_train_anchor(data, anchor_lstm, anchor_optimizer, loss_fn):

    '''main training function 
    iterates over all reviewers and each review session 
    for each student of the review session, 
    '''
    anchor_lstm.train()
    num_epochs = 5
    
    for epoch in range(num_epochs):
        num_decisions = 0
        num_correct = 0
        for review_session in data:
                if review_session.shape[0] < 3:
                    continue 
                hidden_anchor_states = (torch.zeros(1,1,hidden_size).to(device).to(torch.float) ,
                            torch.zeros(1,1,hidden_size).to(device).to(torch.float)) 
                            
                anchor_lstm.zero_grad()

                lstm_input, reviewer_decision = get_input_output_data(review_session, "SVM+Decision")
                preds, hidden, all_hidden = anchor_lstm(lstm_input,hidden_anchor_states)


                preds = preds.squeeze(0).to(device)
                loss_ll = loss_fn(preds, reviewer_decision)
                loss_ll.backward()
                anchor_optimizer.step()

                ### Accuracy ##########################################
                decisions = torch.argmax(preds, dim=1) == reviewer_decision

                correct = decisions.sum().item()
                num_decisions+= len(decisions)
                num_correct += correct
                preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
    return num_correct/num_decisions


def train_anchor(data, anchor_lstm, anchor_optimizer, loss_fn):
    '''main training function 
    iterates over all reviewers and each review session 
    for each student of the review session, 
    '''
    anchor_lstm.train()
    num_epochs = 5
    
    for epoch in range(num_epochs):
        num_decisions = 0
        num_correct = 0
        for review_session in data:
                hidden_anchor_states = (torch.zeros(1,1,hidden_size).to(device).to(torch.float) ,
                            torch.zeros(1,1,hidden_size).to(device).to(torch.float)) 
                            
                anchor_lstm.zero_grad()

                lstm_input, reviewer_decision = get_input_output_data_items(review_session)
                preds, hidden, all_hidden = anchor_lstm(lstm_input,hidden_anchor_states)


                preds = preds.squeeze(0).to(device)
                loss_ll = loss_fn(preds, reviewer_decision)
                loss_ll.backward()
                anchor_optimizer.step()

                ### Accuracy ##########################################
                decisions = torch.argmax(preds, dim=1) == reviewer_decision
                correct = decisions.sum().item()
                num_decisions+= len(decisions)
                num_correct += correct
                preds = torch.argmax(preds, dim=1).cpu().detach().numpy().reshape((preds.shape[0], 1))
    return num_correct/num_decisions

def eval_anchor(data, anchor_lstm, step):
    anchor_lstm.eval()
    num_decisions = 0
    num_correct = 0
    review_sessions = []
    review_sessions_lstm = []
    reviewer_decisions = []
    all_predictions = []
    hidden_states = []
    lstm_inputs = []
    for review_session in data:
            hidden_anchor_states = (torch.zeros(1,1,hidden_size).to(device),
                            torch.zeros(1,1,hidden_size).to(device))

            lstm_input, reviewer_decision = get_input_output_data_items(review_session)
            preds, (anchor, cell_state), all_hidden = anchor_lstm(lstm_input,hidden_anchor_states)

            ### Correlation between all hidden states and the reviewer decisions
            reviewer_decisions.extend(reviewer_decision.cpu().detach().numpy())
            lstm_inputs.extend(lstm_input.squeeze().cpu().detach().numpy())
            hidden_states.extend(all_hidden.squeeze().cpu().detach().numpy())


            preds = preds.squeeze(0)

            ### Accuracy ##########################################
            decisions = torch.argmax(preds, dim=1) == reviewer_decision
            all_predictions.extend(torch.argmax(preds, dim=1).detach().numpy())


            correct = decisions.sum().item()
            all_decisions = len(decisions)
            num_decisions+= all_decisions
            num_correct += correct
            review_session = np.array(review_session)
            preds = torch.argmax(preds, dim=1).cpu().detach().numpy().reshape((preds.shape[0], 1))
            review_sessions.append(np.array(review_session))


            review_session_to_save = np.hstack((review_session[1:], preds))

            review_sessions_lstm.append(np.array(review_session_to_save))

    #print("Correlation between reviewer decisions and hidden states: ", r)
    return num_correct/num_decisions, review_sessions_lstm, hidden_states, reviewer_decisions, all_predictions#, correlation(review_sessions_lstm), review_sessions_lstm


if __name__ == "__main__":
    main()
    





