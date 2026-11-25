# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

Implement a Q-Learning control algorithm using the Gymnasium library to train an agent to learn the optimal path in a FrozenLake environment. The agent should learn the best actions through trial and error while maximizing the total reward and reaching the goal successfully.

## Software Requirements


## Environment Description

The environment used is **FrozenLake-v1** with a **5×5 grid**. The agent starts from the starting state and must reach the goal while avoiding the holes. The environment has **25 states** and **4 possible actions**: Left, Down, Right, and Up. A reward of **1** is given when the agent reaches the goal, while other movements receive a reward of **0**. The Q-Learning algorithm is used to learn the optimal policy for navigating the environment.

## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm

1. Create a **5×5 FrozenLake environment** using the Gymnasium library.
2. Initialize a **Q-table with 25 states and 4 actions** for the environment.
3. Set the learning parameters such as **learning rate (α), discount factor (γ), and exploration rate (ε)**.
4. Select actions using the **epsilon-greedy strategy** to balance exploration and exploitation.
5. Execute the selected action and observe the **next state and reward** from the environment.
6. Update the Q-table using the **Q-Learning update rule** based on the reward and maximum Q-value of the next state.
7. Repeat the process for multiple episodes while gradually reducing the exploration rate.
8. After training, select the action with the **highest Q-value** in each state to obtain the learned policy.
9. Evaluate the agent using the learned policy and plot the **learning curve** based on episode rewards.





## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(num_episodes):
    state, info = env.reset(seed=episode)
    total_reward = 0

    for step in range(100):
        # Choose an action using epsilon-greedy policy
        action = choose_action(state, epsilon)

        # Take the action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning target uses the maximum Q-value
        # of the next state (off-policy learning).
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        # Update Q-value
        Q[state, action] += alpha * (target - Q[state, action])

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Reduce exploration gradually
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# Derive the state-value function and greedy policy
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)

print("Training completed.")
print("Final epsilon:", epsilon)








```
---

## Output

```text

Final Q-table:
[[0.663 0.698 0.698 0.663]
 [0.663 0.735 0.735 0.698]
 [0.698 0.774 0.774 0.735]
 [0.735 0.815 0.735 0.774]
 [0.774 0.    0.735 0.735]
 [0.698 0.735 0.735 0.663]
 [0.698 0.    0.774 0.698]
 [0.735 0.    0.815 0.735]
 [0.774 0.857 0.    0.774]
 [0.    0.    0.    0.   ]
 [0.735 0.774 0.    0.698]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.902 0.902 0.815]
 [0.857 0.95  0.902 0.   ]
 [0.774 0.815 0.815 0.735]
 [0.774 0.857 0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.95  0.95  0.857]
 [0.902 1.    0.95  0.902]
 [0.815 0.815 0.857 0.774]
 [0.815 0.857 0.902 0.815]
 [0.857 0.902 0.95  0.   ]
 [0.902 0.95  1.    0.902]
 [0.    0.    0.    0.   ]]

Estimated State-Value Function:
[[0.698 0.735 0.774 0.815 0.774]
 [0.735 0.774 0.815 0.857 0.   ]
 [0.774 0.    0.    0.902 0.95 ]
 [0.815 0.857 0.    0.95  1.   ]
 [0.857 0.902 0.95  1.    0.   ]]

Learned Policy:
[['D' 'D' 'D' 'D' 'L']
 ['D' 'R' 'R' 'D' 'L']
 ['D' 'L' 'L' 'D' 'D']
 ['D' 'D' 'L' 'D' 'D']
 ['R' 'R' 'R' 'R' 'L']]

Average reward over last 1000 episodes: 0.495 
```
<img width="988" height="627" alt="Screenshot 2026-11-25 141334" src="https://github.com/user-attachments/assets/748ba5a2-56a1-417c-ad58-be1e885a5dbb" />

---

## Result



The Q-Learning algorithm successfully learned a policy for the 5×5 FrozenLake environment. The final Q-table contains learned action values for all 25 states. The estimated state-value function shows higher values near the goal state, with the goal state having a value of 1. The average reward over the last 1000 episodes was **0.495**, indicating approximately **49.5% successful episodes**.




---

## Inference

```text
The results show that Q-Learning can learn an effective navigation policy through trial and error. The learned Q-values increase toward the goal, showing that the agent has learned which actions lead to higher future rewards. The learned policy successfully guides the agent toward the goal while avoiding many of the holes. Although the learning curve appears mostly flat because rewards are sparse, the Q-table and final policy demonstrate successful learning.



```

---

