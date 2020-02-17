
# Banana Gathering Navigation: Report

### 1. Introduction

This file provides a description of the implementation of a smart agent that can navigate in such a way that it collects as many yellow bananas as possible, while avoiding the blue bananas, in a given environement. The report describes the learning algorithm, the chosen hyperparameters, the neural networks architecure and shows the performance of the trained agent as the evolution of the rewards it obtaines over multiple episode runs.

### 2. Implementation

#### 2.1 Learning Algorithm
The learning algorihtm is based on the Deep Q-Network principle, where the hyperparameters have been adapted specifically to this problem.

The algorithm first builds two networks that both map state to action values. Both networks are based on the same architecture, described in the Network Architecture subsection below. The first network is continuously being trained at each step of the event that unfolds. The second network is trained every UPDATE_EVERY time-steps from random (sate, action, reward, next_sate) samples, saved in the agent's memory. These samples are selected from a batch of size BATCH_SIZE from the agent's episodic memory of capacity BUFFER_SIZE.

The second network is used as a target for the Q-function that gives the optimium policy. The first (local) network is optimized to converge towards the second (target) network through the computation of the mean-squared-error (MSE) between their two respective Q-functions. The optimizer of choice is Adam with a learning rate of LR.

The update of the target network is performed via a parameter TAU such that: target_params = TAU*train_params + (1-TAU)*target_params.

The actions are selected mostly at random in the beginning, then progressively following the epsillon-greedy policy dictated by the parameter epsilon which starts at *eps_start* and decays with a rate of *eps_decay* towards its final value *eps_end*.


#### 2.2 Network Architecture
The neuronal network (NN) model was design with two hidden layers of 128 nodes each. The network has an input size of 37, corresponding to the dimension of the state space, while the output size is 4 corresponding to the dimension of the action space. The activation functions between layers are all ReLU. The output layer is linear (no activation functin requried).

#### 2.3 Hyperparameters Optimization

The list of hyperparameters of the current commit is as follows:

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

*eps_start* = 1.0
*eps_end* = 0.01
*eps_decay* = 0.995


### 3. Plot of Rewards

The following plot shows the training evoluiton of the agent, over multiple episodes. The agent learns really fast (by episode 400) the action-value function that generates the optimum policy to receive an average reward (over 100 episodes) of at least +13. The learning stagnates to a reward of around 16 after the 600th episode.

![image](https://github.com/mionescu/udacity-navigation/blob/report_improvement/rewards_plot_v1.png)
*Rewards evolution over multiple episodes*


### 4. Ideas for Future Work

To adapt to the (relatively) large state space of the environment, the hyperparameters need to be more thoroughly optimized.

Currently, there is a large variance on the rewards, which means that even though the average score (over 100 episodes, for instance) is a little high, there is no guarantee that the score for every episode is sufficiently high. In fact, it can even be close to 0. The variance of the model could possibly be further improved by changing a) number of nodes in the NN and/or b) the number of layers in the NN model.

In trying to improve the performance of the model, the number of nodes in the two hidden layers of the network was increased from 128 to 512. In this case, an average score of 13.11 has been obtained by episode 700, much later than in the inital design with two hidden layers of 128 nodes. However, the improvement brought by the increased number of hidden nodes is that the variance on the scores was greatly reduced.

The epsillon, and tau parameters could be influencing the stability of the DQN in converging towards an optimum policy. The fact that there is a plateau in rewards around 15-16 bananas, could be due to eps_end factor. Possibly, *eps_end* should be much smaller to reduce even further the randomness of the agent.

To improve the agent's performance, some ideas for future work are the implementation of a double DQN, a dueling DQN, and/or prioritized experience replay!
