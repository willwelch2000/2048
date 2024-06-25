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

### Notes
- Without a small alpha value, the training can become unstable, leading to huge weight values. 
- The alpha of the deep q-learning system is scaled to the average rewards of the system