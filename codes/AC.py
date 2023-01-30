import torch
import numpy as np
import torch.nn.functional as F

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
        is_exploratory = action != np.argmax(logits.detach().cpu().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())

class critic_net(torch.nn.Module):
    def __init__(self,nS):
        super(critic_net, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=nS, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
        )
        self.v_value = torch.nn.Sequential(torch.nn.Flatten(),
                                        torch.nn.Linear(3136, 512),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(512, 1)) 
        
    def forward(self, state):
        x = self.feature_extract(state)
        x = self.v_value(x)
        return x

class AC():
    def __init__(self, cfg):
        self.env = cfg.env
        
        self.gamma = cfg.model.gamma
        self.policy_model_fn  = cfg.model.policy_model_fn 
        self.policy_optimizer_fn  = cfg.model.policy_optimizer_fn 
        self.policy_optimizer_lr = cfg.model.policy_optimizer_lr

        self.critic_model_fn  = cfg.model.critic_model_fn 
        self.critic_optimizer_fn  = cfg.model.critic_optimizer_fn 
        self.critic_optimizer_lr = cfg.model.critic_optimizer_lr
        
        self.nS = cfg.model.nS  
        self.nA = cfg.model.nA
        self.epochs = cfg.model.epochs  

    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_history, self.exploratory_history = [], []

        self.policy_model = self.policy_model_fn(self.nS, self.nA).to(self.device)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, 
                                                       self.policy_optimizer_lr)
        
        self.critic_model = self.critic_model_fn(self.nS).to(self.device)
        self.critic_optimizer = self.critic_optimizer_fn(self.critic_model, 
                                                       self.critic_optimizer_lr)

        self.epochs_completed = 0
        pass

    def optimize_model(self, state, reward, next_state, action, done, log_p):
        state = torch.from_numpy(state).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)
        done = torch.tensor(done).int().to(self.device)

        # _, _, log_p, entropy = self.policy_model.full_pass(state[np.newaxis, ...])
        q_s = self.critic_model(state[np.newaxis, ...])
        q_ss = self.critic_model(next_state[np.newaxis, ...])

        TD_error = reward + (self.gamma * q_ss * (1 - done) - q_s)
        
        critic_loss = F.smooth_l1_loss(q_s, TD_error)
        policy_loss = -log_p * TD_error.detach() 
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 

        self.policy_optimizer.zero_grad()
        policy_loss.backward() 
        self.policy_optimizer.step()

        self.policy_loss.append(policy_loss.item())
        self.critic_loss.append(critic_loss.item())
        
    def interaction_step(self, state):
        action, is_exploratory, logpa, _ = self.policy_model.full_pass(state)
        self.exploratory.append(is_exploratory)
        return action, logpa
    
    def epoch(self, runing_bar):
        self.rewards, self.exploratory = [], []
        state = self.env.reset().__array__()
        self.policy_loss, self.critic_loss = [], []
        
        while True:
            action, logpa = self.interaction_step(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            next_state, reward, done, info = self.env.step(action)
            next_state = next_state.__array__()
            self.optimize_model(state, reward, next_state, action, done, logpa)
            
            state = next_state
            self.rewards.append(reward)
            
            if done or info["flag_get"]:
                break
            
        self.reward_history.append(np.sum(self.rewards))
        self.exploratory_history.append(np.mean(self.exploratory))
        
        self.epochs_completed += 1
        
        if self.epochs_completed > 30 and self.epochs_completed % 10 == 0:
            runing_bar.set_postfix(reward=np.mean(self.reward_history[self.epochs_completed-30:self.epochs_completed]),
                                   exploratory=np.mean(self.exploratory_history[self.epochs_completed-30:self.epochs_completed]),
                                   p_loss=np.mean(self.policy_loss),
                                   c_loss=np.mean(self.critic_loss),
                                   )