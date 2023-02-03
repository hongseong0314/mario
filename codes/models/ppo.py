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
        np_logits = logits.detach().cpu().numpy()
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        np_actions = actions.detach().cpu().numpy()
        logpas = dist.log_prob(actions)
        np_logpas = logpas.detach().cpu().numpy()
        is_exploratory = np_actions != np.argmax(np_logits, axis=1)
        return np_actions, np_logpas, is_exploratory

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def get_predictions(self, states, actions):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        logpas = dist.log_prob(actions)
        entropies = dist.entropy()
        return logpas, entropies

    def select_greedy_action(self, states):
        logits = self.forward(states)
        return np.argmax(logits.detach().squeeze().cpu().numpy())

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

class PPO():
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
        self.policy_optimization_epochs = cfg.model.policy_optimization_epochs
        self.value_optimization_epochs = cfg.model.value_optimization_epochs

        self.tau = cfg.model.tau
        self.batch_size = cfg.model.batch_size

        self.policy_optimization_epochs = cfg.model.policy_optimization_epochs
        self.policy_model_max_grad_norm = cfg.model.policy_model_max_grad_norm
        self.policy_clip_range = cfg.model.policy_clip_range
        self.policy_stopping_kl = cfg.model.policy_stopping_kl

        self.value_optimization_epochs = cfg.model.value_optimization_epochs
        self.value_model_max_grad_norm = cfg.model.value_model_max_grad_norm
        self.value_stopping_mse = cfg.model.value_stopping_mse

        self.entropy_loss_weight = cfg.model.entropy_loss_weight

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

    def optimize_model(self):
        self.p_losses, self.c_losses = [], []
        
        states = torch.from_numpy(self.s_m).to(self.device)
        actions = torch.from_numpy(self.a_m).to(self.device)
        logpas = torch.tensor(self.logp).to(self.device)
        gaes = torch.tensor(self.gaes).to(self.device)
        done = torch.tensor(done).int().to(self.device)

        values = torch.tensor(self.q_as).to(self.device)
        returns = torch.tensor(self.returns).to(self.device)

        for _ in range(self.policy_optimization_epochs):
            batch_idxs = np.random.choice(len(self.r_m), self.batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            logpas_batch = logpas[batch_idxs]

            old_logpas, old_entropy = self.policy_model.get_predictions(states_batch,
                                                                        actions_batch)
            ratios = (old_logpas - logpas_batch).exp()
            pi_obj = gaes_batch * ratios

            pi_obj_clipped = gaes_batch * ratios.clamp(1.0 - self.policy_clip_range,
                                                       1.0 + self.policy_clip_range)
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -old_entropy.mean() * self.entropy_loss_weight

            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 
                                           self.policy_model_max_grad_norm)
            self.policy_optimizer.step()

            self.p_losses.append((policy_loss + entropy_loss).item())

            with torch.no_grad():
                logpas_pred_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (logpas - logpas_pred_all).mean()
                if kl.item() > self.policy_stopping_kl:
                    break
            
        for _ in range(self.value_optimization_epochs):
            batch_idxs = np.random.choice(len(self.r_m), self.batch_size, replace=False)
            states_batch = states[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]

            values_pred = self.value_model(states_batch)
            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range, 
                                                                                    self.value_clip_range)
            v_loss = (returns_batch - values_pred).pow(2)
            v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
            critic_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 
                                        self.value_model_max_grad_norm)
            self.value_optimizer.step()

            with torch.no_grad():
                values_pred_all = self.value_model(states)
                mse = (values - values_pred_all).pow(2).mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    break
            
            self.c_losses.append(critic_loss.item())
                

        self.policy_loss.append(np.mean(self.p_losses))
        self.critic_loss.append(np.mean(self.c_losses))
    
    def epoch(self, runing_bar):
        self.s_m, self.r_m, self.a_m, self.logp, self.gae = [], [], [], [], []
        self.q_as, self.q_sas = [], []
        self.exploratory = []
        state = self.env.reset().__array__()
        self.policy_loss, self.critic_loss = [], []
        
        while True:
            action, logpas, is_exploratory = self.policy_model.full_pass(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            next_state, reward, done, info = self.env.step(action[0])
            next_state = next_state.__array__()
            
            q_a = self.critic_model(torch.from_numpy(state)[np.newaxis, ...].to(self.device))
            q_sa = self.critic_model(torch.from_numpy(next_state)[np.newaxis, ...].to(self.device))

            self.s_m.append(state) 
            self.r_m.append(reward) 
            self.a_m.append(action[0]) 
            self.logp.append(logpas) 
            self.exploratory.append(is_exploratory) 
            self.q_as.append(q_a.detach().cpu().numpy())
            self.q_sas.append(q_sa.detach().cpu().numpy())

            state = next_state 

            if done or info["flag_get"]:
                break

        self.get_gae()

        self.reward_history.append(np.sum(self.r_m))
        self.exploratory_history.append(np.mean(self.exploratory))
        self.epochs_completed += 1
        
        if self.epochs_completed > 30 and self.epochs_completed % 10 == 0:
            runing_bar.set_postfix(reward=np.mean(self.reward_history[self.epochs_completed-30:self.epochs_completed]),
                                   exploratory=np.mean(self.exploratory_history[self.epochs_completed-30:self.epochs_completed]),
                                   p_loss=np.mean(self.policy_loss),
                                   c_loss=np.mean(self.critic_loss),)
    def get_gae(self):
        T = len(self.r_m)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        self.returns = np.array([np.sum(discounts[:T-t] * self.r_m[t:]) for t in range(T)])
        self.td_targets = np.zeros_like(self.r_m)
        self.gae = np.zeros_like(self.r_m)
        gae_cumulative = 0
        V_ss = 0

        for t in reversed(range(0, T)):
            delta = self.r_m[t] + self.gamma * V_ss - self.q_as[t]
            gae_cumulative = self.gamma * self.tau * gae_cumulative + delta
            self.gae[t] = gae_cumulative
            V_ss = self.q_as[t]
            self.td_targets[t] = self.gae[t] + self.q_as[t]