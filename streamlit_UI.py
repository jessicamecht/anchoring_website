import streamlit as st
import numpy as np 
import pandas as pd 
import SessionState as session_state
from rl_anchoring.models import anchor_models, actor_critic_models
from rl_anchoring.action_selection import *
import math
import torch
import os, os.path
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heuristic = False

def generate_random_code():
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    return ''.join(random.choice(letters) for i in range(10))

def display_thank_you(code, button_placeholder_1, button_placeholder_2, placeholder, explanation_placeholder_2, explanation_placeholder):
    st.write("Thanks for contributing to this study.")
    st.write('Please submit the following code to MTurk:')
    st.title(code)
    placeholder.empty()
    button_placeholder_1.empty()
    button_placeholder_2.empty()
    explanation_placeholder_2.empty()
    explanation_placeholder.empty()

def save_data(last_decisions, shown_instances, path, code):
    last_decisions = np.array(last_decisions)
    last_decisions = last_decisions.reshape(last_decisions.shape[0],1)

    data = np.append(shown_instances, last_decisions, axis=1)
    df = pd.DataFrame(data)
    df.to_csv(f'./data/data_ai/mturk_review_session_data_{path}_{code}.csv')
  
def main():         
    count = len([name for name in os.listdir('./data/data_ai/')])
  
    action_idx = None
    data_paths = ['books_reviews.csv', 'books_reviews_2.csv', 'books_reviews_3.csv', 
    'books_reviews_4.csv', 'books_reviews_5.csv', 'books_reviews_6.csv', 
    'books_reviews_7.csv', 'books_reviews_8.csv','books_reviews_9.csv', 
    'books_reviews_10.csv', 'books_reviews_11.csv', 'books_reviews_12.csv', 
    'books_reviews_13.csv', 'books_reviews_14.csv', 'books_reviews_15.csv']
    possible_idxs = [10, 11, 12]
    random_idx = random.choice(possible_idxs)
    print(random_idx)
    #random.randrange(len(data_paths))

    print('random', random_idx, count)
    path = data_paths[random_idx]
    df = pd.read_csv(f"./review_sessions/{path}")[0:50]

    possible_next_instances = np.array(df)
    input_size = 2
    hidden_size = state_size = 1
    hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1, hidden_size).to(device)) #initial anchor 
    possible_next_instances_mask = np.ones(50)
    possible_next_instances_mask[0] = 0

    if len(possible_next_instances) < len(possible_next_instances_mask):
        possible_next_instances_mask[len(possible_next_instances):] = 0

    review_length = 50

    anchor_lstm = anchor_models.AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./rl_anchoring/state_dicts/anchor_lstm_items_all_unbalanced_1.pt', map_location=device))

    actor = actor_critic_models.Actor(state_size, review_length).to(device)
    actor.load_state_dict(torch.load('./rl_anchoring/state_dicts/actor_model_1.pt', map_location=device))

    #### Init session state to preserve states 
    sess_state = session_state.SessionState(progress=0, action_idx=0, last_decisions=[], possible_next_instances_mask=possible_next_instances_mask, shown_instances=[])
    ss = session_state.get(progress=0, action_idx=0, last_decisions=[], possible_next_instances_mask=possible_next_instances_mask, shown_instances=[])
    
    st.title('Book Reviews')

    ####Declare placeholders for visuals 
    explanation_placeholder = st.empty()
    explanation_placeholder.write(f'Consider the following rating of a book from another user. \
    In total, you will see {df.shape[0]} individual book review texts. Please consider every review individually. \
    Every review is from a different book.\
    ')
    explanation_placeholder_2 = st.empty()
    explanation_placeholder_2.write('Please indicate if you think you\'d like to read the book after reading the review from the other user.')
    
    placeholder = st.empty()
    next_instance = df.loc[ss.action_idx]
    placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])
    button_placeholder_1 = st.empty()
    button_placeholder_2 = st.empty()

    if button_placeholder_1.button("Yes, I'd like to read the book."):
        ss.last_decisions.append(1)
        lstm_input = torch.tensor(np.array(ss.last_decisions, dtype=float)).to(device).to(torch.float)
        
        if np.array(ss.possible_next_instances_mask).sum() != 0:
            if not heuristic:
                action_idx = get_next_action(lstm_input, ss.shown_instances, hidden_anchor_state, ss.possible_next_instances_mask, actor, anchor_lstm)
            else:
                action_idx = heuristic_select_next_action(1, possible_next_instances, ss.possible_next_instances_mask)
        
            ss.action_idx = action_idx
            next_instance = df.loc[ss.action_idx]
            placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])

    if button_placeholder_2.button("No, I'd NOT like to read the book."):
        ss.last_decisions.append(0)
        lstm_input = torch.tensor(np.array(ss.last_decisions, dtype=float)).to(device).to(torch.float)
        
        if np.array(ss.possible_next_instances_mask).sum() != 0:
            if not heuristic:
                action_idx = get_next_action(lstm_input, ss.shown_instances, hidden_anchor_state, ss.possible_next_instances_mask, actor, anchor_lstm)
            else:
                action_idx = heuristic_select_next_action(0, possible_next_instances, ss.possible_next_instances_mask)
        
            ss.action_idx = action_idx
            next_instance = df.loc[ss.action_idx]
            placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])


    if np.array(ss.possible_next_instances_mask).sum() == 0 :
        code = generate_random_code()
        display_thank_you(code, button_placeholder_1, button_placeholder_2, placeholder, explanation_placeholder_2, explanation_placeholder)
        save_data(ss.last_decisions, ss.shown_instances, path, code)
    
    tmp_mask = ss.possible_next_instances_mask
    tmp_mask[ss.action_idx] = False
    ss.possible_next_instances_mask = tmp_mask
    ss.shown_instances.append(next_instance)

if __name__ == "__main__":
    main()