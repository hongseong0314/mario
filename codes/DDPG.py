import torch
import numpy as np
import torch.nn.functional as F

class policy_net(torch.nn.Module):
    def __init__(self,nS, nA, out_activation_fc=F.tanh):
        super(policy_net, self).__init__()
        self.out_activation_fc = out_activation_fc
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
        return self.out_activation_fc(x)


class critic_net(torch.nn.Module):
    def __init__(self,nS, nA):
        super(critic_net, self).__init__()

        self.feature_extract = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=nS, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        self.v_value = torch.nn.Sequential(
                                    torch.nn.Linear(3136+nA, 512),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(512, 1)) 
        
    def forward(self, state, action):
        x = self.feature_extract(state)
        x = torch.cat((x, action), dim=1)
        x = self.v_value(x)
        return x

class DDPG():
    def __init__(self, cfg):
        self.env = cfg.env
        
        self.gamma = cfg.model.gamma
        self.policy_model_fn  = cfg.model.policy_model_fn 
        self.policy_optimizer_fn  = cfg.model.policy_optimizer_fn 
        self.policy_optimizer_lr = cfg.model.policy_optimizer_lr

        self.critic_model_fn  = cfg.model.critic_model_fn 
        self.critic_optimizer_fn  = cfg.model.critic_optimizer_fn 
        self.critic_optimizer_lr = cfg.model.critic_optimizer_lr

        self.replay_buffer_fn=cfg.model.replay_buffer_fn
        
        self.nS = cfg.model.nS  
        self.nA = cfg.model.nA
        self.epochs = cfg.model.epochs  
        self.batch_size = cfg.model.batch_size
        self.freq  = 1
        self.start_size = cfg.model.batch_size * 3
        self.tau = cfg.model.tau

    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_history, self.exploratory_history = [], []

        self.target_policy_model = self.policy_model_fn(self.nS, self.nA).to(self.device)
        self.online_policy_model = self.policy_model_fn(self.nS, self.nA).to(self.device)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, 
                                                       self.policy_optimizer_lr)
        
        self.target_critic_model = self.critic_model_fn(self.nS, self.nA).to(self.device)
        self.online_critic_model = self.critic_model_fn(self.nS, self.nA).to(self.device)
        self.critic_optimizer = self.critic_optimizer_fn(self.online_critic_model, 
                                                       self.critic_optimizer_lr)
        self.update_networks(tau=1.0)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.action_fn()

        self.epochs_completed = 0
        self.step_count = 0
        pass

    def optimize_model(self):
        s, a, r, s_prime, done = self.replay_buffer.sample(self.batch_size)

        s = torch.from_numpy(s).to(self.device)
        r = torch.tensor(r).to(self.device)
        a = torch.tensor(a, dtype=torch.float32).to(self.device)
        s_prime = torch.from_numpy(s_prime).to(self.device)
        done = torch.tensor(done).int().to(self.device)

        # expected Q
        argmax_a_q_sp = self.target_policy_model(s_prime)
        max_a_q_sp = self.target_critic_model(s_prime, argmax_a_q_sp)
        target_q_sa = r + self.gamma * max_a_q_sp * (1 - done)

        # critic loss 
        q_sa = self.online_critic_model(s, a)
        critic_loss = F.smooth_l1_loss(q_sa, target_q_sa.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 

        # policy loss
        argmax_a_q_s = self.online_policy_model(s)
        policy_loss = -self.online_critic_model(s, argmax_a_q_s).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward() 
        self.policy_optimizer.step()

        self.policy_loss.append(policy_loss.item())
        self.critic_loss.append(critic_loss.item())

        
    def interaction_step(self, state):
        return self.training_strategy.select_action(self.online_policy_model, 
                                                      state,) + np.random.normal(loc=0, scale=0.1, size=self.nA) # explore를 위한 noise 추가
    
    def epoch(self, runing_bar):
        self.rewards, self.exploratory, self.policy_loss, self.critic_loss = [], [], [], []
        state = self.env.reset().__array__()
        
        while True:
            if len(self.replay_buffer) > self.start_size:
                self.optimize_model()

                if self.step_count % self.freq == 0 and self.epochs_completed != 0:
                    self.update_networks()
 
            action = self.interaction_step(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            next_state, reward, done, info = self.env.step(np.argmax(action))
            next_state = next_state.__array__()
            
            self.replay_buffer.put(([state], action, reward, [next_state], done))
            
            state = next_state
            self.rewards.append(reward)
            
            self.step_count += 1
            
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
            
    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_critic_model.parameters(), 
                                  self.online_critic_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(), 
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)