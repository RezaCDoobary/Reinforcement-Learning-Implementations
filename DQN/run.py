import src as navigation
import gym
import numpy as np
from src.agent import deque, torch, optim
from src.model import device
import matplotlib.pyplot as plt
import os
import environments


def dqn(env, agent, n_episodes=2000, max_t=1000, solved_at = 200.0, stop_at_goal = True):
    scores = []
    mean_scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    #eps = eps_start                    # initialize epsilon
    solved_in = None

    date_and_time_string = date_and_time()
    folder_destination = "results/" + date_and_time_string
    
    if_folder_not_there_create(folder_destination)
    new_folder = folder_number(folder_destination)
    
    folder_destination = folder_destination+"/"+new_folder

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done,_= env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        agent.policy.update(i_episode)
        mean_scores.append(np.mean(scores_window))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if stop_at_goal:
            if np.mean(scores_window)>=solved_at:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.local_model.state_dict(), folder_destination + "/checkpoint.pth")
                solved_in = i_episode-100
                break

    torch.save(agent.local_model.state_dict(), folder_destination + "/checkpoint_final.pth")
    plot(scores, mean_scores, folder_destination + '/plot.png')    
    return scores, solved_in, mean_scores, folder_destination



def json_to_dict(filename):
    import json
    with open(filename, "r") as read_file:
        data = json.load(read_file)
    return data

def date_and_time(with_time = False):
    """
    Returns a YYYYMMDD string with out without time
    
    Params
    ======
        with_time (boolean) : Gives time in string is True, does not otherwise.
    """
    import datetime
    
    YEAR = str(datetime.datetime.today().year)
    
    MONTH = str(0) + str(datetime.datetime.today().month) if datetime.datetime.today().month < 10 \
    else str(datetime.datetime.today().month)
    
    DAY = str(0) + str(datetime.datetime.today().day) if datetime.datetime.today().day < 10 \
    else str(datetime.datetime.today().day)
    
    HOUR = str(0) + str(datetime.datetime.today().hour) if datetime.datetime.today().hour < 10\
    else str(datetime.datetime.today().hour)
    
    MINUTE = str(0) + str(datetime.datetime.today().minute) if datetime.datetime.today().minute < 10\
    else str(datetime.datetime.today().minute)
    
    if with_time:
        return YEAR+MONTH+DAY+'_'+HOUR+MINUTE
    else:
        return YEAR+MONTH+DAY
    
    

        
def if_folder_not_there_create(folder_destination):
    """Creates a folder if it does not exist
    
    Params
    ======
        folder_destination (string) : The path to be created if it does not already exist.
    """
    if not os.path.exists(folder_destination):
        os.makedirs(folder_destination)
        
def folder_number(path):
    """Creates a folder of the form run_# in case we have multiple runs per day
    
    Params
    ======
        path (string) : The path to be created if it does not already exist.
    """

    folders = []

    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for folder in d:
            folders.append(os.path.join(r, folder))

    numbers = []
    for f in folders:
        ff = f.split('_')
        numbers.append(ff[-1])

    if len(numbers) == 0:
        new_folder = "run"+"_1"
    else:
        m = int(numbers[-1])
        new_folder = "run"+"_"+str(m+1)

    if_folder_not_there_create(path + "//" +new_folder)  
    return new_folder

def write_to_json(data, filename):
    import json
    with open(filename, "w") as write_file:
        json.dump(data, write_file, indent = 4)


def plot(scores, ma_scores, folder):
    ax = plt.subplots(figsize=(20, 10))
    plt.plot(np.arange(len(scores)), scores,label = 'Scores')
    plt.plot(np.arange(len(scores)), ma_scores, label = 'Moving average scores')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc='upper left')
    plt.savefig(folder)
    plt.show()


def train_model(filename = 'input.json'):

    config = json_to_dict(filename)

    BUFFER_SIZE = config['BUFFER_SIZE'] # replay buffer size
    BATCH_SIZE = config['BATCH_SIZE']        # minibatch size
    GAMMA = config['GAMMA']           # discount factor
    TAU = config['TAU']              # for soft update of target parameters
    LR = config['LR']            # learning rate 
    UPDATE_EVERY = config['UPDATE_EVERY']        # how often to update the network
    hidden_layers = config["HIDDEN_LAYERS"]
    drop_p = config["DROP_P"]
    with_dueling = bool(config["WITH_DUELING"])
    isDDQN = bool(config["WITH_DDQN"])

    n_episodes = config["N_EPISODES"]
    max_t = config["MAX_T"]
    eps_start = config["EPS_START"]
    eps_end = config["EPS_END"]
    eps_decay = config["EPS_DECAY"]
    stop_at_goal = bool(config["STOP_AT_GOAL"])
    solved_at = config["SOLVED"]


    env1 = environments.environment_gym(config['ENV'])
    env = environments.env_wrapper(env1)
    seed = 0



    action_size = env.action_size()
    state_size = len(env.reset())
    memory = navigation.ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    local_model = navigation.QModelNet(state_size, action_size, seed,  \
        hidden_layers  = hidden_layers, drop_p = drop_p, dueling = with_dueling).to(device)
    target_model = navigation.QModelNet(state_size, action_size, seed,  \
        hidden_layers  = hidden_layers, drop_p = drop_p, dueling = with_dueling).to(device)
    optimiser = optim.Adam

    
    eps = navigation.epsilon(eps_start, eps_decay, eps_end)
    policy = navigation.epsilon_greedy(eps, 1)

    agent = navigation.Agent( state_size=state_size, 
                              action_size=action_size,
                              seed=0,
                              memory=memory,
                              local_model=local_model,
                              target_model=target_model,
                              optimiser = optimiser,
                              policy = policy,
                              UPDATE_EVERY = UPDATE_EVERY,
                              GAMMA = GAMMA,
                              BATCH_SIZE = BATCH_SIZE,
                              LR = LR,
                              TAU = TAU,
                              isDDQN = isDDQN)

    scores, solved_in, mean_scores, folder_destination = dqn(env, agent,n_episodes, max_t,solved_at,stop_at_goal)
    write_to_json(config, folder_destination + '/input.json')
    return scores, solved_in, mean_scores


scores, solved_in, mean_scores = train_model()


