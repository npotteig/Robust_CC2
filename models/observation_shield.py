import torch
from torch import nn
from torch.optim import Adam
from torch.distributions.categorical import Categorical

from models.memory_buffer import PPOMemory

class ObservationShieldActor(nn.Module):
    def __init__(self,
                 next_state_label_dim,
                 output_label_dim,
                 lr=0.001
                 ):
        super(ObservationShieldActor, self).__init__()
        self.node_shield = nn.Linear(next_state_label_dim + 1, output_label_dim)
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, input):
        node_output = self.node_shield(input)
        # Create node-wise categorical distributions
        return Categorical(logits=node_output)

class ObservationShieldCritic(nn.Module):
    def __init__(self,
                 next_state_label_dim,
                 lr=0.001
                 ):
        super(ObservationShieldCritic, self).__init__()
        self.node_shield = nn.Linear(next_state_label_dim + 1, 1)
        self.optimizer = Adam(self.parameters(), lr=lr)
        
    def forward(self, input):
        if input.dim() == 2:
            input = input.unsqueeze(0)
        node_output = self.node_shield(input)
        # Sum node outputs to get a single value estimate
        return node_output.sum(dim=1, keepdim=True)

class ObservationShieldPPOAgent:
    def __init__(self,
                 next_state_label_dim,
                 output_label_dim,
                 gamma=0.99,
        lmbda=0.95,
        clip=0.1,
        bs=5,
        epochs=6,
        a_kwargs=dict(),
        c_kwargs=dict(),
        training=True,
    ):
        self.actor = ObservationShieldActor(next_state_label_dim, output_label_dim, **a_kwargs)
        self.critic = ObservationShieldCritic(next_state_label_dim, **c_kwargs)
        self.memory = PPOMemory(bs)

        self.args = (next_state_label_dim, output_label_dim)
        self.kwargs = (gamma, lmbda, clip, bs, epochs, a_kwargs, c_kwargs, training)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.clip = clip
        self.bs = bs
        self.epochs = epochs

        self.training = training
        self.deterministic = False
        self.mse = nn.MSELoss()
    
    def set_deterministic(self, val):
        self.deterministic = val
    
    def train(self):
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()
    
    def save(self, outf="saved_models/ppo.pt"):
        me = (self.args, self.kwargs)

        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "agent": me,
            },
            outf,
        )
    
    def remember(self, s, a, v, p, r, t):
        self.memory.remember(s, a, v, p, r, t)
    
    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
    
    @torch.no_grad()
    def get_action(self, input_labels):
        # x,ei = x_ei
        distro = self.actor(input_labels)

        # I don't know why this would ever be called
        # during training, but just in case, putting the
        # logic block outside the training check
        if self.deterministic:
            action = distro.probs.argmax(dim=-1)
        else:
            action = distro.sample()

        if not self.training:
            return action.item()

        value = self.critic(input_labels)
        prob = distro.log_prob(action)
        return action.numpy(), value.item(), prob.numpy()

    def learn(self, verbose=True):
        for e in range(self.epochs):
            s, a, v, p, r, t, batches = self.memory.get_batches()
            
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(r), reversed(t)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + self.gamma * discounted_reward
                rewards.insert(0, discounted_reward)

            r = torch.tensor(rewards, dtype=torch.float)
            r = (r - r.mean()) / (r.std() + 1e-5)  # Normalize rewards

            advantages = r - torch.tensor(v)

            for b_idx, b in enumerate(batches):
                b = b.tolist()
                new_probs = []

                s_ = [s[idx] for idx in b]
                a_ = [a[idx] for idx in b]
                batched_states = s_
                dist = self.actor(*batched_states)

                critic_vals = self.critic(*batched_states)
                new_probs = dist.log_prob(torch.tensor(a_))
                old_probs = torch.tensor([p[i] for i in b])
                entropy = dist.entropy()

                a_t = advantages[b]

                # Equiv to exp(new) / exp(old) b.c. recall: these are log probs
                r_theta = (new_probs - old_probs).exp()
                clipped_r_theta = torch.clip(
                    r_theta, min=1 - self.clip, max=1 + self.clip
                )

                # Use whichever one is minimal
                actor_loss = torch.min(r_theta * a_t, clipped_r_theta * a_t)
                actor_loss = -actor_loss.mean()

                # Critic uses MSE loss between expected value of state and observed
                # reward with discount factor
                critic_loss = self.mse(r[b].unsqueeze(-1), critic_vals)

                # Not totally necessary but maybe will help?
                entropy_loss = entropy.mean()

                # Calculate gradient and backprop
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss
                self.__zero_grad()
                total_loss.backward()
                self.__step()

                if verbose:
                    print(
                        f"[{e}] C-Loss: {0.5 * critic_loss.item():0.4f}  A-Loss: {actor_loss.item():0.4f} E-loss: {-entropy_loss.item() * 0.01:0.4f}"
                    )

        # After we have sampled our minibatches e times, clear the memory buffer
        self.memory.clear()
        return total_loss.item()
    
    def __zero_grad(self):
        self.actor.opt.zero_grad()
        self.critic.opt.zero_grad()

    def __step(self):
        self.actor.opt.step()
        self.critic.opt.step()