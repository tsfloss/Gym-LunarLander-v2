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
