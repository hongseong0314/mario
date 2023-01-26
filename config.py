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