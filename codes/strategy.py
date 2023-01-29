import numpy as np
import torch

class GreedyStrategy():
    def __init__(self):
        pass
    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy()
        return greedy_action

class EGreedyLinearStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.001, decay_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self):
        epsilon = 1 - self.t / self.decay_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().numpy().squeeze()

        if np.random.rand() >= self.epsilon:
            action = np.argmax(q_values)
        else: 
            action = np.random.randint(len(q_values))
        
        self.epsilon = self._epsilon_update()
        return action