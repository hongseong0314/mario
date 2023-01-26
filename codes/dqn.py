import torch
import torch.nn.functional as F
import numpy as np

class Qnet(torch.nn.Module):
    def __init__(self, nS, nA):
        super(Qnet, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=nS, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(),
                                        torch.nn.Linear(3136, 512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(512, nA))    
        # self.initialize_weights()
    
    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, torch.nn.Linear):
    #             torch.nn.init.kaiming_uniform_(m.weight)
    #             torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature_extract(x)
        x = self.classifier(x)
        return x

class DQN():
    def __init__(self, cfg):
        self.env = cfg.env
        
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.nA = cfg.model.nA
        self.epochs = cfg.model.epochs  
        self.start_size = cfg.model.start_size
        self.freq = cfg.model.freq 

    def optimize_model(self):
        s, a, r, s_prime, done = self.replay_buffer.sample(self.batch_size)
        
        s = torch.tensor(s).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r).to(self.device)
        s_prime = torch.tensor(s_prime).to(self.device)
        done = torch.tensor(done).to(self.device)

        q_a = self.online_model(s).gather(1,a)

        with torch.no_grad():
            max_a_q_sp = self.target_model(s_prime).detach().max(1)[0].unsqueeze(1)
            target_q_sa = r + (self.gamma * max_a_q_sp * (1 - done.float()))
        
        loss = F.smooth_l1_loss(q_a, target_q_sa)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        self.losses.append(loss.item())
        
    def interaction_step(self, state):
        action = self.training_strategy.select_action(self.online_model, state)
        return action
    
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_history, self.loss_history = [], []

        self.online_model = self.value_model_fn(self.nS, self.nA).to(self.device)
        self.target_model = self.value_model_fn(self.nS, self.nA).to(self.device)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)


        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0
        pass
    
    def epoch(self, runing_bar):
        while True:
            flag = next(self.stepper)
            if flag:
                break
        self.epochs_completed += 1
        
        if self.epochs_completed > 30 and self.epochs_completed % 10 == 0:
            runing_bar.set_postfix(reward=np.mean(self.reward_history[self.epochs_completed-30:self.epochs_completed]),
                                   loss=np.mean(self.loss_history[self.epochs_completed-30:self.epochs_completed]),)
            
    def step_run(self):
        self.cum_cost, self.losses = 0, []
        state = self.env.reset().__array__()
        while(True):
            if len(self.replay_buffer) > self.start_size and self.step_count % 3 == 0:
                self.optimize_model()

            if self.step_count % self.freq == 0 and self.epochs_completed != 0:
                self.update_network()
 
            action = self.interaction_step(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            
            next_state, reward, done, info = self.env.step(action)
            next_state = next_state.__array__()
            
            self.cum_cost += reward

            self.replay_buffer.put(([state], action, reward, [next_state], done))
            
            state = next_state
            self.step_count += 1

            if done or info["flag_get"]:
                done = True
                self.reward_history.append(self.cum_cost)
                self.loss_history.append(np.mean(self.losses))
                self.cum_cost, self.losses = 0, []
                state = self.env.reset().__array__()

            yield done