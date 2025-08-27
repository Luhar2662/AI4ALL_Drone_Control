import torch
import torch.nn as nn
import torch.nn.functional as F

def encode_local_state(agent_pos, goal_pos, grid, range=1):
    """
    Constructs a feature vector:
    - Flattened local grid centered on agent, with visual range 'range'
    - Agent position
    - Goal position

    Args:
        agent_pos: Tuple (row, col) of agent
        goal_pos: Tuple (row, col) of goal
        grid: 2D numpy array of the environment, with:
          - empty space as 0
          - boundaries and obstacles as -1
          - goal as positive 1
        obs_range: Number of grid cells to look around the agent

    Returns:
        state_vector: 1D numpy array
    """
    rows, cols = grid.shape
    r, c = agent_pos

    # Crop local region
    local = grid[r-range:r+range+1, c-range:c+range+1]
    local_flat = local.flatten()  # shape: (local_size^2,)

    # Normalize positions
    agent_norm = [r / rows, c / cols]
    goal_r, goal_c = goal_pos
    goal_norm = [goal_r / rows, goal_c / cols]

    # Combine features
    state_vector = np.concatenate([local_flat, agent_norm, goal_norm]).astype(np.float32)
    return state_vector


class DQN(nn.module):

  def __init__(self, input_size, num_actions):
    super(DQN, self).__init__()
    self.model = nn.Sequential(
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions)
    )

  def forward(self, x):
    return self.model(x)

class ReplayBuffer():

  #format: state, action, next_state, reward, done

  def __init__(self, capacity = 10000):
    self.buffer = deque(maxlen=capacity)

  def add(self, state, action, next_state, reward, done):
    self.buffer.append(state, action, next_state, reward, done)

  def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done),
        )

  def __len__(self):
        return len(self.buffer)

def train_step(policy_net, target_net, opt, replay_buffer, batch_size, gamma):
  if len(replay_buffer) < batch_size:
    return

  state, action, reward, next_state, done = replay_buffer.sample(batch_size)

  q_values = policy_net(state)
  next_q_values = target_net(next_state)

  q_val = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
  next_q_val = next_q_values.max(1)[0]
  expected_q_val = reward + gamma * next_q_val * (1-done)

  loss = F.mse_loss(q_val, expected_q_val.detatch())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  return loss.item()


def train_dqn(
      env, 
      agent,
      episodes=500,
      batch_size=32,
      gamma=.99,
      epsilon_start=1.0,
      epsilon_end=.1,
      epsilon_decay=.995,
      lr=1e-3,
      replay_capacity=10000,
      target_update_freq=10,
      obs_range=2
      ):
   
   actions = agent.actions
   input_size = (2*obs_range + 1)**2 + 4
   num_actions = len(actions)

   #initialize networks for DQN:
   policy_net = DQN(input_size, num_actions)
   target_net = DQN(input_size, num_actions)

   #sync networks
   target_net.load_state_dict(policy_net.state_dict())
   target_net.eval()

   optimizer = optim.Adam(policy_net.parameters, lr = lr)

   #initialize replay buffer and epsilon value (will be scheduled)
   replay_buffer = ReplayBuffer(replay_capacity)

   epsilon = epsilon.start

   for ep in range(episodes):
      agent.reset()
      # reset the environment for this run, with RANDOM obstacles
      # TODO: write env.random_reset, and set up the "grid" so that flattening works correctly
      env.reset_random()
      goal_pos = env.get_goal_position()

      total_reward = 0
      losses = []

      #step through episode!
      for step in range(env.episode_steps):
         state = encode_local_state(agent.position, goal_pos, env.grid, obs_range)
         state_tensor = torch.FloatTensor([state])

         if random.random() < epsilon:
            action = random.randint(0, num_actions - 1)
         else: 
            with torch.no_grad():
               #choose according to our current policy!
               q_vals = policy_net(state_tensor)



