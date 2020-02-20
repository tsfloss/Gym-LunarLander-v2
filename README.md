# Gym-LunarLander-v2
My solution to solving the Gym Environment LunarLander-v2 (Discrete) using Deep Q Learning.

The code is written to be executed in an IPython console.

## Training
Once the code is executed the model can be trained for a number of training_episodes by:
```
train(training_episodes)
```
The model trains until all episodes have passed or until the problem is considered solved, when 100 subsequent episodes have a mean reward (rolling mean reward) of 200. Once done, the model will be saved to a file.

## Testing
Once the model is trained or a saved model is preset. It can be tested for a number of test_episodes by:
```
test(filename, test_episodes, render_every)
```
where render_every sets how often the environment should be rendered.

A pretrained model (see graph below) has been included and can be tested.

## Technical Information
The environment is solved using a Deep Q Learning implementation. Due to the immense state space, conventional Q learning using a Q value table is not feasible. Hence we use a neural network to predict future actions and train the net using the Q learning algorithm.

The model starts out with a lot of random actions as a means of exploration (epsilon decay). Every couple of actions it trains the neural network using the rewards and Q learning algorithm. 

The neural net consists of an input layer of 512 nodes that takes the 8 input values from the environment using relu activation, followed by a hidden layer of 256 nodes with relu activation. The Output layer outputs 4 values, one for each action, using linear activation.

The most optimal hyperparameters were found to be:
```
learning_rate = 0.001
epsilon = 1.0
epsilon_decay = 0.995
gamma = 0.99
```

After a bit more than 250 episodes the model starts to learn how to land. At slightly less than 600 episodes it has solved the enivornment and can land pretty much all the time.

The graph of the rewards shows a continuous learning process:
![RewardGraph](/trained_model/LunarLander-graph.png)

