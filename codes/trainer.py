import torch
import pickle
import os

from tqdm import tqdm
from config import dqn_config, rein_config, ac_config, ddpg_config, ppo_config
from codes.strategy import EGreedyLinearStrategy, GreedyStrategy
from codes.buffer import ReplayBuffer_list
from codes.env import get_mario_env
from codes.utills import create_directory

def trainer(name):
    if name == 'DQN':
        from codes.dqn import Qnet, DQN
        cfg = dqn_config()
        cfg.model.batch_size = 32 
        cfg.model.training_strategy_fn = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
                                            min_epsilon=0.001, 
                                            decay_steps=20000)

        cfg.model.value_model_fn = lambda nS, nA: Qnet(nS, nA)
        cfg.model.value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.model.replay_buffer_fn = lambda : ReplayBuffer_list(buffer_limit=100000, batch_size=cfg.model.batch_size)
        cfg.env = get_mario_env()

        agent = DQN(cfg)
        agent.setup()
        
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}.pth".format(agent.__class__.__name__, 
                                                            cfg.model.training_strategy_fn().__class__.__name__, 
                                                            cfg.model.batch_size,  
                                                            )
        torch.save(agent.online_model.state_dict(), online_model_name)

        create_directory("history")
        history = {
                "reward":agent.reward_history,
            }
        history_path = 'history/result_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                            cfg.model.training_strategy_fn().__class__.__name__, 
                                                            cfg.model.batch_size, 
                                                            )
        with open(history_path,'wb') as f:
            pickle.dump(history,f)
    
    elif name == 'REINFORCE':
        from codes.reinforce import policy_net, REINFORCE
        cfg = rein_config()
        cfg.model.policy_model_fn = lambda nS, nA : policy_net(nS, nA)
        cfg.model.policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.env = get_mario_env()

        agent = REINFORCE(cfg)
        agent.setup()
      
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        create_directory("weight")                
        online_model_name = "weight/q_{}.pth".format(agent.__class__.__name__,)
        torch.save(agent.online_model.state_dict(), online_model_name)

        create_directory("history")
        history = {
            "reward":agent.reward_history,
            }
        history_path = 'history/result_{}.pkl'.format(agent.__class__.__name__,)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)
    
    elif name == 'AC':
        from codes.AC import critic_net, policy_net, AC
        cfg = ac_config()
        cfg.model.policy_model_fn = lambda nS, nA : policy_net(nS, nA)
        cfg.model.policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.model.critic_model_fn = lambda nS : critic_net(nS)
        cfg.model.critic_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.env = get_mario_env()

        agent = AC(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        create_directory("weight")                
        online_model_name = "weight/q_{}.pth".format(agent.__class__.__name__,)
        torch.save(agent.online_model.state_dict(), online_model_name)

        create_directory("history")
        history = {
            "reward":agent.reward_history,
            }
        history_path = 'history/result_{}.pkl'.format(agent.__class__.__name__,)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)
    
    elif name == 'ddpg':
        from codes.DDPG import policy_net, critic_net, DDPG
        cfg = ddpg_config()

        cfg.model.training_strategy_fn = GreedyStrategy

        cfg.model.policy_model_fn = lambda nS, nA : policy_net(nS, nA)
        cfg.model.policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.model.critic_model_fn = lambda nS, nA : critic_net(nS, nA)
        cfg.model.critic_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.env = get_mario_env()
        cfg.model.tau = 0.005
        cfg.model.batch_size = 32 
        cfg.model.replay_buffer_fn = lambda : ReplayBuffer_list(buffer_limit=100000, batch_size=cfg.model.batch_size)

        agent = DDPG(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        create_directory("weight")                
        online_model_name = "weight/q_{}.pth".format(agent.__class__.__name__,)
        torch.save(agent.online_model.state_dict(), online_model_name)

        create_directory("history")
        history = {
            "reward":agent.reward_history,
            }
        history_path = 'history/result_{}.pkl'.format(agent.__class__.__name__,)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

    elif name == 'ppo':
        from codes.ppo import policy_net, critic_net, PPO
        cfg = ppo_config()
        cfg.model.policy_model_fn = lambda nS, nA : policy_net(nS, nA)
        cfg.model.policy_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.model.critic_model_fn = lambda nS : critic_net(nS)
        cfg.model.critic_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
        cfg.env = get_mario_env()

        agent = PPO(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        create_directory("weight")                
        online_model_name = "weight/q_{}.pth".format(agent.__class__.__name__,)
        torch.save(agent.online_model.state_dict(), online_model_name)

        create_directory("history")
        history = {
            "reward":agent.reward_history,
            }
        history_path = 'history/result_{}.pkl'.format(agent.__class__.__name__,)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)