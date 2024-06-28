# 2048
A C# project that has programs for playing 2048, incorporating machine learning

## Game2048
This is the class library that actually implements the 2048 game.
- It includes an interface for a class that can display a 2048 game and a concrete implementation of the interface: a command-line display. 

## Run2048
This is a simple program that lets the user control a game via the command line. 

## AI2048
This is the class that incorporates AI into the 2048 game. 

Q-Learning is implemented for 2048 in a few different forms
- Approximate Q-Learning uses features of the game, extracted via functions defined in the agent, to approximate the q-values
- Deep Q-Learning uses a neural network to approximate the q-values. The input layer of the network is defined in the agent and isn't necessarily just the tile values in the game. For example, several values are included to show a mapping of which squares are equal

### Example Successful Setup Specifications
Neural Network
- 4 total layers (2 middle layers)
- Input size of 256
- 50 middle nodes
- First two layer transforms use ReLUWithSlope for activation (0.1 positive slope, 0.001 negative slope)
- Final layer transform uses NoActivation for activation (linear)

Deep Q Learner specifications
- Starting epsilon of 1, epsilon decay of 0.99, minimum epsilon of 0.1
- Iterations before net transfer = 1000
- Starting alpha of 0.000001
- Alpha is recalculated after each 100 episodes to be 0.01 / average rewards of last 100

Results (average rewards in 100 rounds, all are far above random)
- v1: 2508.6
- v2: 2497.68
- v3: 2577.8
- v4: 2579.72
- v5: 2655.32
- v6: 2517.4
- v7: 2465.16
- v8: 2474.48
- v9: 2568.92

### Notes
- Without a small alpha value, the training can become unstable, leading to huge weight values. 
- The alpha of the deep q-learning system is scaled to the average rewards of the system
