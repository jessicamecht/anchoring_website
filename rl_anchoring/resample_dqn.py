import torch 
import random
from models.dqn_models import * 
from utils import * 
from sklearn.model_selection import KFold
from models.anchor_models import AnchorLSTM
import math
from plot import * 

steps_done = 0
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 100

def select_action(state, policy_net, possible_next_instances_mask):
    assert(possible_next_instances_mask.sum() != 0)
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            logits = policy_net(state)
            idx = (logits * possible_next_instances_mask).max(1)[1].view(1, 1)
            if not possible_next_instances_mask[idx] and possible_next_instances_mask.sum != 0:
                idx = (possible_next_instances_mask).argmax().unsqueeze(-1).unsqueeze(-1)
            return idx
    else:
        #pick random possible action
        possible_idxs = torch.nonzero(possible_next_instances_mask)
        pick_idx = random.randrange(len(possible_idxs))
        return possible_idxs[pick_idx].unsqueeze(-1)

def anchor_reward(anchors):
    '''try to minimize the anchor'''
    return (1 - torch.abs(anchors)).sum()


def main(hidden_size, n_iters=1):
    ### Load data #######################
    data = load_data_items()
    svm_predictions, svm_confidence, features, target_decision, final_decision, Item_Number, reviewer_score = data[0][0]
    kf = KFold(n_splits=5)

    input_size = 2
    state_size = hidden_size
    anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./state_dicts/anchor_lstm_items_all_unbalanced_{hidden_size}_new.pt', map_location=torch.device('cpu')))
    all_resampled_review_sessions = []
    split = 0
    for train_index, test_index in kf.split(data):
        split+=1
        data_train = data[train_index]
        data_test = data[test_index]
        resampled_review_sessions = train_resampling(data_train, anchor_lstm, hidden_size, hidden_size)
        resampled_eval = eval_net(data)
        all_resampled_review_sessions.extend(resampled_eval)

        generate_plot(all_resampled_review_sessions, f"confidence_items_resampled_{split}")
    generate_plot(all_resampled_review_sessions, f"all_final_confidence_items_resampled")

def eval_net(data):

    #### INIT MODELS ########################################################################
    state_size, input_size, hidden_size = 1, 2, 1
    policy_net = DQN(state_size, 50).to(device)
    policy_net.load_state_dict(torch.load(f"./state_dicts/policy_model.pt", map_location=device))
    anchor_lstm = AnchorLSTM(input_size, hidden_size).to(device)
    anchor_lstm.load_state_dict(torch.load(f'./state_dicts/anchor_lstm_items_all_unbalanced_1.pt', map_location=device))
    all_review_sessions = []

    for possible_next_instances in data:
        possible_next_instances_mask = torch.ones(50).to(device)#.to(torch.long)

        if len(possible_next_instances) < len(possible_next_instances_mask): # if we have a shorter sequence, pad 
            possible_next_instances_mask[len(possible_next_instances):] = 0

        assert(possible_next_instances_mask.sum() == len(possible_next_instances))

        action_sequence = []
        state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1, hidden_size).to(device)) #initial anchor 
        action_idx = select_action(state[0].squeeze(-1), policy_net, possible_next_instances_mask)
        action = possible_next_instances[action_idx]
        possible_next_instances_mask[action_idx] = False
        assert(possible_next_instances_mask.sum() == len(possible_next_instances)-1)
        action_sequence.append(action)

        while possible_next_instances_mask.sum() != 0:
            action_idx = select_action(state[0].squeeze(-1), policy_net, possible_next_instances_mask)                    
            action = possible_next_instances[action_idx]
            possible_next_instances_mask[action_idx] = False
            action_sequence.append(action)

            lstm_input, reviewer_decision = get_input_output_data_items(np.array(action_sequence))
            with torch.no_grad():
                predictions, state, _ = anchor_lstm(lstm_input,state)

        all_review_sessions.append(action_sequence)
    assert(len(all_review_sessions) == len(data))
    return all_review_sessions

def train_resampling(data, anchor_lstm, state_size = 1, hidden_size = 1, steps_done = 200, TARGET_UPDATE=10):

    action_size = 50

    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)

    num_episodes = 1
    for i_episode in range(num_episodes):
        print('i_episode', i_episode)
        # Initialize the environment and state
        n_steps_since_last_pos = 0 
        state = 0
        all_resampled = []
        for t, possible_next_instances in enumerate(data):
            possible_next_instances_mask = torch.ones(50).to(device)#.to(torch.long)

            if len(possible_next_instances) < len(possible_next_instances_mask): # if we have a shorter sequence, pad 
                possible_next_instances_mask[len(possible_next_instances):] = 0

            assert(possible_next_instances_mask.sum() == len(possible_next_instances))

            action_sequence = []
            state = (torch.zeros(1,1,hidden_size).to(device), torch.zeros(1,1, hidden_size).to(device)) #initial anchor 
            action_idx = select_action(state[0].squeeze(-1), policy_net, possible_next_instances_mask)
            action = possible_next_instances[action_idx]
            possible_next_instances_mask[action_idx] = False
            assert(possible_next_instances_mask.sum() == len(possible_next_instances)-1)
            action_sequence.append(action)

            while possible_next_instances_mask.sum() != 0:
                # Select and perform an action
                action_idx = select_action(state[0].squeeze(-1), policy_net, possible_next_instances_mask)
                done =  possible_next_instances_mask.sum() == 0
                    
                action = possible_next_instances[action_idx]
                possible_next_instances_mask[action_idx] = False
                action_sequence.append(action)

                # Observe new state
                lstm_input, reviewer_decision = get_input_output_data_items(np.array(action_sequence))
                with torch.no_grad():
                    predictions, next_state, _ = anchor_lstm(lstm_input,state)
                (next_hidden, next_cell_state) = next_state
                reward = anchor_reward(next_hidden)
                reward = torch.tensor([reward], device=device)

                # Store the transition in memory
                memory.push(state[0].squeeze(-1), action_idx, next_state[0].squeeze(-1), reward)

                # Move to the next state
                state = next_state
            assert(possible_next_instances_mask.sum() == 0)
            assert(len(action_sequence) == len(possible_next_instances))

            # Perform one step of the optimization (on the policy network)
            optimize_model(memory, policy_net, target_net, optimizer)
            all_resampled.append(action_sequence)
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(), f"./state_dicts/policy_model.pt")
    torch.save(target_net.state_dict(), f"./state_dicts/target_model.pt")
    return all_resampled

def optimize_model(memory, policy_net, target_net, optimizer, BATCH_SIZE=10, GAMMA = 0.999):
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)

    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

if __name__ == "__main__":
    main(1)
