import torch 
import numpy as np 
from utils import * 
import random 
from action_selection import * 
from models.actor_critic_models import * 
import torch.optim as optim
from models.models import AnchorLSTM
import math
import matplotlib.pyplot as plt
import collections
from sklearn.model_selection import KFold
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps_start = 0.9
eps_end = 0.0
eps_decay = 200

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def anchor_reward(anchors):
    return (1 - torch.abs(anchors)).sum()

def select_action(output, eps_end, eps_start, eps_decay, steps_done):
    '''selects the next action to be executed based on the current student shown
    actions can be 0: rejection, 1:admission
    '''
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done/eps_decay)
    #exploration/exploitation trade-off
    if sample > eps_threshold:
        with torch.no_grad():
            return output.max(2)[1].view(1)
    else:
        return torch.tensor([[random.randrange(len(output))]], device=device, dtype=torch.long)

def train_resampling(data, anchor_lstm):
    '''trains the resampling with actor critic method given possible next instances
    param: data
    param: anchor_lstm '''

    anchor_lstm.eval()

    state_size = 1
    hidden_size = 1
    steps_done = 200

    actor = Actor(state_size, 50).to(device)
    critic = Critic(state_size).to(device)

    actor.train()
    critic.train()

    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())

    review_sessions = []


    for possible_next_instances in data:
        possible_next_instances_mask = np.ones(50)
        if len(possible_next_instances) < len(possible_next_instances_mask): # if we have a shorter sequence
            possible_next_instances_mask[len(possible_next_instances):] = 0
    
        while np.array(possible_next_instances_mask).sum() > 0:
            hidden_anchor_state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1, hidden_size).to(device)) #initial anchor 
            length_of_sequence = min(random.randint(3,30), len(possible_next_instances))
            instance_sequence = []
            instance = possible_next_instances[0]
            possible_next_instances_mask[0] = False
            instance_sequence.append(instance)

            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            for i in range(length_of_sequence):
                steps_done+=1
                lstm_input, reviewer_decision = get_input_output_data_items(np.array(instance_sequence))
                print(lstm_input.shape, reviewer_decision.shape)

                with torch.no_grad():
                    predictions, (state, _), _ = anchor_lstm(lstm_input,hidden_anchor_state)
                #get probability for action and critic given anchor 
                output, value = actor(state[:,-1:,:]), critic(state[:,-1:,:])#only feed the last state 
                #eliminate impossible actions (already sampled students and students from a different year)
                valid_output = output * torch.tensor(possible_next_instances_mask).to(device)
                #Select student to be sampled
                valid_output = valid_output[:,:,0:len(possible_next_instances)]
                action_idx = select_action(valid_output,eps_end, eps_start, eps_decay, steps_done).item()
                next_instance = possible_next_instances[action_idx]

                instance_sequence.append(next_instance)
                #remove the student s.t. he can't be sampled again  
                possible_next_instances_mask[action_idx] = False

                # get reward (min sum of anchor)
                reward = anchor_reward(state)
                rewards.append(reward)
                done = possible_next_instances_mask.sum()==0
                log_prob = torch.log(output.squeeze()[action_idx])
                logp = torch.log2(output.squeeze())
                entropy = (-output.squeeze()*logp).sum()
                entropy += entropy.mean()
                
                log_probs.append(log_prob.unsqueeze(0))
                values.append(value.squeeze(2))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))
            review_sessions.append(instance_sequence)
            next_state = state
            next_value = critic(next_state)
                
            returns = compute_returns(next_value, rewards, masks)
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
                
            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            optimizerA.zero_grad()
            optimizerC.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            optimizerA.step()
            optimizerC.step()

    #correlation(review_sessions)
    #plot_n_decisions_vs_confidence(review_sessions, f"./figures/resampled_confidence_at_end_of_training_{input_for_lstm}_{fold}.png")
    torch.save(actor.state_dict(), "./state_dicts/actor_model.pt")
    torch.save(critic.state_dict(), "./state_dicts/critic_model.pt")
    return review_sessions
    
    

def main(n_iters=1):
    ### Load data #######################
    data = load_data_items()
    svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score = data[0][0]
    kf = KFold(n_splits=5)

    input_size = 2
    hidden_size = 1
    state_size = 1
    anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./state_dicts/anchor_lstm_SVM+Decision.pt', map_location=torch.device('cpu')))
    all_resampled_review_sessions = []

    for train_index, test_index in kf.split(data):
        data_train = data[train_index]
        data_test = data[test_index]
        resampled_review_sessions = train_resampling(data_train, anchor_lstm)
        all_resampled_review_sessions.extend(resampled_review_sessions)

    generate_plot(all_resampled_review_sessions, f"./all_final_confidence_items_resampled")

    #actor.load_state_dict(torch.load(anchor_path))


if __name__ == "__main__":
    main()
