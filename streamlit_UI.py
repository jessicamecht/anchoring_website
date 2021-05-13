import streamlit as st
import numpy as np 
import pandas as pd 
import SessionState as session_state
from rl_anchoring.models import models, actor_critic_models
import math
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_action(output):
    print(output.shape, output.max(2)[1].view(1))
    return torch.tensor(output.max(2)[1].view(1), dtype=torch.int64, device=device)


def get_next_action(lstm_input, hidden_anchor_state, possible_next_instances_mask, actor, anchor_lstm):
    actor.eval()
    anchor_lstm.eval()
    with torch.no_grad():
        redictions, (state, _), _ = anchor_lstm(lstm_input,hidden_anchor_state)
        output = actor(state[:,-1:,:])
        #eliminate impossible actions (already sampled students and students from a different year)
        valid_output = output * torch.tensor(possible_next_instances_mask).to(device)
        #Select student to be sampled
        action_idx = select_action(valid_output).item()
    return action_idx
        
def main():            
    action_idx = None

    data_paths = ['books_reviews.csv', 'books_reviews_2.csv', 'books_reviews_3.csv', 'books_reviews_4.csv',
     'books_reviews_4.csv', 'books_reviews_5.csv', 'books_reviews_6.csv', 'books_reviews_7.csv']
    df = pd.read_csv(f"./review_sessions/{data_paths[1]}")

    possible_next_instances = np.array(df)
    input_size = hidden_size = state_size = 1
    hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1, hidden_size).to(device)) #initial anchor 
    possible_next_instances_mask = np.ones(50)

    if len(possible_next_instances) < len(possible_next_instances_mask):
        possible_next_instances_mask[len(possible_next_instances):] = 0

    review_length = 50

    anchor_lstm = models.AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./rl_anchoring/state_dicts/anchor_lstm_items.pt'))

    actor = actor_critic_models.Actor(state_size, review_length).to(device)
    actor.load_state_dict(torch.load('./rl_anchoring/state_dicts/actor_model.pt'))
    sess_state = session_state.SessionState(progress=0, action_idx=0, last_decisions=[0], possible_next_instances_mask=possible_next_instances_mask)
    
    ss = session_state.get(progress=0, action_idx=0, last_decisions=[0], possible_next_instances_mask=possible_next_instances_mask)
    st.title('Book Reviews')
    explanation_placeholder = st.empty()
    explanation_placeholder.write('Consider the following rating of a book from another user. \
    In total, you will see 50 individual book review texts. Please consider every review individually. \
    Every review is from a different book.\
    ')
    explanation_placeholder_2 = st.empty()
    explanation_placeholder_2.write('Please indicate if you think you\'d like to read the book after reading the review from the other user.')
    
    placeholder = st.empty()
    placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])
    button_placeholder = st.empty()
    button_placeholder_2 = st.empty()
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789'
    
    if button_placeholder.button("Yes, I'd like to read the book."):
        ss.last_decisions = ss.last_decisions + [1]
        lstm_input = torch.tensor(np.array(ss.last_decisions, dtype=float)).to(device).to(torch.float)
        action_idx = get_next_action(lstm_input, hidden_anchor_state, ss.possible_next_instances_mask, actor, anchor_lstm)
        ss.action_idx = action_idx
        placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])
        ss.progress = ss.progress + 1
        tmp_mask = ss.possible_next_instances_mask
        tmp_mask[action_idx] = False
        ss.possible_next_instances_mask = tmp_mask
        if np.array(ss.possible_next_instances_mask).sum() == 0 :
            code = ''.join(random.choice(letters) for i in range(10))
            st.title("Thanks for contributing to this study.")
            st.write('Please submit the following code to MTurk:')
            st.title(code)
            button_placeholder.empty()
            button_placeholder_2.empty()
            placeholder.empty()
            explanation_placeholder_2.empty()
            explanation_placeholder.empty()
            np.save(f'mturk_review_data_{code}.npy', np.array(ss.last_decisions))



    if button_placeholder_2.button("No, I'd NOT like to read the book."):
        ss.last_decisions = ss.last_decisions + [0]
        lstm_input = torch.tensor(np.array(ss.last_decisions, dtype=float)).to(device).to(torch.float)
        action_idx = get_next_action(lstm_input, hidden_anchor_state, ss.possible_next_instances_mask, actor, anchor_lstm)
        ss.action_idx = action_idx
        placeholder.table(df[["summary", "reviewText"]].loc[ss.action_idx])
        ss.progress = ss.progress + 1
        tmp_mask = ss.possible_next_instances_mask
        tmp_mask[action_idx] = False
        ss.possible_next_instances_mask = tmp_mask
        if np.array(ss.possible_next_instances_mask).sum() == 0 :
            st.write("Thanks for contributing to this study.")
            st.write('Please submit the following code to MTurk:')
            code = ''.join(random.choice(letters) for i in range(10))
            st.title(code)
            button_placeholder.empty()
            placeholder.empty()
            button_placeholder_2.empty()
            explanation_placeholder_2.empty()
            explanation_placeholder.empty()
            np.save(f'./data/mturk_review_session_data_{code}.npy', np.array(ss.last_decisions))

        

if __name__ == "__main__":
    main()