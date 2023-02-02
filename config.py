from ml_collections import config_dict

def dqn_config():
    cfg = config_dict.ConfigDict()

    cfg.model = config_dict.ConfigDict()
    cfg.model.value_optimizer_lr = 0.00025
    cfg.model.gamma = 0.90
    cfg.model.nS = 4
    cfg.model.nA = 2
    cfg.model.freq = 1000
    cfg.model.epochs = 40000
    cfg.model.start_size = 1e4
    return cfg

def reinforce_config():
    cfg = config_dict.ConfigDict()

    cfg.model = config_dict.ConfigDict()
    cfg.model.policy_optimizer_lr = 0.0005
    cfg.model.gamma = 0.99
    cfg.model.nS = 4
    cfg.model.nA = 2
    cfg.model.epochs = 1000
    return cfg

def ddpg_config():
    cfg = config_dict.ConfigDict()

    cfg.model = config_dict.ConfigDict()
    cfg.model.policy_optimizer_lr = 0.00005
    cfg.model.critic_optimizer_lr = 0.00005
    cfg.model.gamma = 0.90
    cfg.model.nS = 4
    cfg.model.nA = 2
    cfg.model.epochs = 2000
    return cfg

def ppo_config():
    cfg = config_dict.ConfigDict()

    cfg.model = config_dict.ConfigDict()
    cfg.model.policy_optimizer_lr = 0.00005
    cfg.model.critic_optimizer_lr = 0.00025
    cfg.model.gamma = 0.90
    cfg.model.nS = 4
    cfg.model.nA = 2
    cfg.model.epochs = 2000

    cfg.model.tau = 0.97
    cfg.model.batch_size = 32 

    cfg.model.policy_optimization_epochs = 20
    cfg.model.policy_model_max_grad_norm = float('inf')
    cfg.model.policy_clip_range = 0.1 
    cfg.model.policy_stopping_kl = 0.02

    cfg.model.value_optimization_epochs = 20
    cfg.model.value_model_max_grad_norm = float('inf')
    cfg.model.value_stopping_mse = 25

    cfg.model.entropy_loss_weight = 0.01
    return cfg

