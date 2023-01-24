import torch
import pickle
from tqdm import tqdm
from codes.strategy import EGreedyLinearStrategy
from config import config
from codes.model import DQN, Qnet
from codes.env import get_mario_env
from codes.buffer import ReplayBuffer_list
from codes.utills import create_directory
def run():
    cfg = config()
    cfg.model.training_strategy_fn = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
                                        min_epsilon=0.001, 
                                        decay_steps=20000)
    cfg.model.value_model_fn = lambda nS, nA: Qnet(nS, nA)
    cfg.model.value_optimizer_fn = lambda net, lr: torch.optim.Adam(net.parameters(), lr=lr)
    cfg.model.replay_buffer_fn = lambda : ReplayBuffer_list(buffer_limit=100000, batch_size=cfg.model.batch_size)
    cfg.model.batch_size = 32 
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

if __name__ == "__main__":
    run()