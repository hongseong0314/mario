import torch
import numpy as np

class policy_net(torch.nn.Module):
    def __init__(self,nS, nA):
        super(policy_net, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=nS, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.policy_layer = torch.nn.Sequential(torch.nn.Flatten(),
                                        torch.nn.Linear(3136, 512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(512, nA)) 
        
    def forward(self, state):
        x = self.feature_extract(state)
        x = self.policy_layer(x)
        return x

    def full_pass(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())

class REINFORCE():
    def __init__(self, cfg):
        self.env = cfg.env
        
        self.gamma = cfg.model.gamma
        self.policy_model_fn  = cfg.model.policy_model_fn 
        self.policy_optimizer_fn  = cfg.model.policy_optimizer_fn 
        self.policy_optimizer_lr = cfg.model.policy_optimizer_lr
        self.nS = cfg.model.nS  
        self.nA = cfg.model.nA
        self.epochs = cfg.model.epochs  

    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])
       
        discounts = torch.FloatTensor(discounts).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        self.logpas = torch.cat(self.logpas)
        policy_loss = -(discounts * returns * self.logpas).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
    def interaction_step(self, state):
        action, is_exploratory, logpa, _ = self.policy_model.full_pass(state)
        next_state, reward, done, info = self.env.step(action)
        
        self.logpas.append(logpa)
        self.rewards.append(reward)

        return next_state, done or info["flag_get"]
    
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_history, self.logpas_history = [], []

        self.policy_model = self.policy_model_fn(self.nS, self.nA).to(self.device)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, 
                                                       self.policy_optimizer_lr)

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
            runing_bar.set_postfix(reward=np.mean(self.reward_history[self.epochs_completed-30:self.epochs_completed]),)
            
    def step_run(self):
        self.logpas, self.rewards = [], []
        state = self.env.reset().__array__()
        while(True):
            next_state, done = self.interaction_step(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            next_state = next_state.__array__()
            
            state = next_state
            self.step_count += 1

            if done:
                self.optimize_model()
                self.reward_history.append(np.sum(self.rewards))
                self.logpas_history.append(self.logpas)
                self.logpas, self.rewards = [], []
                state = self.env.reset().__array__()

            yield done