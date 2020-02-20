import gym
import numpy as np
from keras.backend import clear_session
from keras.models import Sequential
from keras.activations import relu, linear
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.models import load_model
import random
from collections import deque

class DQNAgent:
    def __init__(self, env, lr, gamma, epsilon, epsilon_dec):
        self.env = env
        self.counter = 0
        self.learning_rate = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.train_rewards = []
        self.deck = deque(maxlen=500000)
        self.batch_size = 32
        self.epsilon_low = 0.01
        self.steps_per_fit = 5
        self.model = self.initialize_model()
    
    def initialize_model(self):
        clear_session()
        model = Sequential()
        model.add(Dense(512, input_dim=self.env.observation_space.shape[0], activation=relu))
        model.add(Dense(256, activation=relu))
        model.add(Dense(self.env.action_space.n, activation=linear))
        model.compile(loss=mean_squared_error, optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model
                
    def save(self, name):
        print(f"Saving model to {name}")
        self.model.save(name)
        
    def update_epsilon(self):
        if self.epsilon > self.epsilon_low:
            self.epsilon *= self.epsilon_dec

    def get_action(self,current_state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.env.action_space.n)
        else:
            return np.argmax(self.model.predict(current_state)[0])
    def DQNfit(self):
        #check buffer length
        if len(self.deck) < self.batch_size:
            return
        #make sample and appropriate lists
        sample = np.array(random.sample(self.deck, self.batch_size))
        current_states = np.squeeze(np.array([i[0] for i in sample]))
        actions = np.array([i[1] for i in sample])
        rewards = np.array([i[2] for i in sample])
        new_states = np.squeeze(np.array([i[3] for i in sample]))
        done_list = np.array([i[4] for i in sample])
        #update Q value using prediction of model
        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(new_states),axis=1))*(1-done_list)
        targets_old = self.model.predict_on_batch(current_states)
        indexes = np.array([i for i in range(self.batch_size)])
        targets_old[[indexes], [actions]] = targets
        #fit model to updated Q value
        self.model.fit(current_states,targets_old,verbose=0,epochs=1)
        
    def train(self,episodes):
        for episode in range(1,episodes+1):
            #reset values for episode
            episode_reward = 0
            current_state = env.reset().reshape(1,-1)
            #run a maximum of steps per episode
            for step in range(1000):
                #render condition
#                if episode % 100 == 0:
#                    env.render()
                #get action from model or random
                action = self.get_action(current_state)
                new_state, reward, done, inf = env.step(action)
                new_state = new_state.reshape(1,-1)
                #update reward and buffer
                episode_reward += reward
                self.deck.append((current_state, action, reward, new_state, done))
                current_state = new_state
                #fit the model every couple of steps, with early stopping protection
                if step % self.steps_per_fit == 0 and np.mean(self.train_rewards[-10:]) < 180:
                    self.DQNfit()
                #if episode is done quit the episode
                if done:
                    break
            #calculate means over last 100 episodes
            self.train_rewards.append(episode_reward)
            last_rewards_mean = np.mean(self.train_rewards[-100:])
            #print episode results and means
            print(f"Episode {episode} \t Reward {round(episode_reward)} \t Epsilon {round(self.epsilon*100)} \t Average Reward {round(last_rewards_mean)} \t Steps {step}")
            #decay epsilon
            self.update_epsilon()
            #if sufficient score is reached break regardless of episode
            if last_rewards_mean > 200:
                print(f"Training completed after {episode} episodes!")
                break
        print("All training episodes done!")
        self.save(f"LunarLander-{last_rewards_mean}.h5")
                
def test_trained_model(env,trained_model,test_episodes):
    test_rewards = []
    for episode in range(1,test_episodes+1):
        #reset values for episode
        episode_reward = 0
        current_state = env.reset().reshape(1,-1)
        #run a maximum of steps per episode
        for step in range(1000):
            #render condition
            env.render()
            #get action from model or random
            action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, inf = env.step(action)
            new_state = new_state.reshape(1,-1)
            #update reward and buffer
            episode_reward += reward
            current_state = new_state
            if done:
                break
        test_rewards.append(episode_reward)
        last_rewards_mean = np.mean(test_rewards)
        print(f"Episode {episode} \t Reward {round(episode_reward)} \t Average Reward {round(last_rewards_mean)} \t Steps {step}")
    print("Testing completed!")
def train():
    # setting up parameters
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    training_episodes = 300
    #Instantiate Agent
    model = DQNAgent(env, lr, gamma, epsilon, epsilon_decay)
    #Train model
    model.train(training_episodes)
def test():
    test_episodes=3
    model_file = "LunarLander--1.549633858779646.h5"
    trained_model = load_model(model_file)
    test_trained_model(env,trained_model,test_episodes)
    
env = gym.make('LunarLander-v2')
env.seed(22)
np.random.seed(22)
