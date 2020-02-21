import gym
import numpy as np
from keras.backend import clear_session
from keras.models import Sequential
from keras.activations import relu, linear
from keras.layers import Dense
import matplotlib.pyplot as plt
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
        self.rolling_means = []
        self.deck = deque(maxlen=250000)
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

    def take_action(self,current_state):
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
        dones = np.array([i[4] for i in sample])
        #update Q value using prediction of model
        target_updates = rewards + (1-dones) * self.gamma * (np.amax(self.model.predict_on_batch(new_states),axis=1))
        targets_current = self.model.predict_on_batch(current_states)
        targets_current[[i for i in range(self.batch_size)], [actions]] = target_updates
        
        #fit model to updated Q value
        self.model.fit(current_states,targets_current,verbose=0,epochs=1)
        
    def train(self,episodes):
        for episode in range(1,episodes+1):
            #reset values for episode
            episode_reward = 0
            current_state = self.env.reset().reshape(1,-1)
            #run a maximum of steps per episode
            for step in range(1000):
                #render condition
                if (episode - 1) % 100 == 0:
                    self.env.render()
                #get action from model or random
                action = self.take_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.reshape(1,-1)
                #update reward and buffer
                episode_reward += reward
                self.deck.append((current_state, action, reward, new_state, done))
                current_state = new_state
                #if reward is already low before crashing, stop episode
                if episode_reward < -200:
                    break
                #fit the model every step_per_fit steps, with early stopping protection
                if step % self.steps_per_fit == 0 and np.mean(self.train_rewards[-10:]) < 180:
                    self.DQNfit()
                #if episode is done quit the episode
                if done:
                    break
            #calculate means over last 100 episodes
            self.train_rewards.append(episode_reward)
            self.rolling_means.append(np.mean(self.train_rewards[-100:]))
            #print episode results and means
            print(f"Episode {episode} \t Reward {round(episode_reward)} \t Epsilon {round(self.epsilon*100)} \t Average Reward {round(self.rolling_means[-1])} \t Steps {step}")
            #decay epsilon
            self.update_epsilon()
            #if sufficient score is reached break regardless of episode
            if self.rolling_means[-1] > 200:
                print(f"Training completed after {episode} episodes!")
                break
        print("All training episodes done!")
        #save model to file
        self.save(f"LunarLander-{episode}.h5")
        plt.plot(self.train_rewards,label = "Episode Reward")
        plt.plot(self.rolling_means,label = "Rolling Mean Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.axhline(y=0,color = 'black')
        plt.legend()
        plt.show()
                
def test_trained_model(env,trained_model,test_episodes, render_every):
    test_rewards = []
    for episode in range(1,test_episodes+1):
        #reset values for episode
        episode_reward = 0
        current_state = env.reset().reshape(1,-1)
        #run a maximum of steps per episode
        for step in range(1000):
            #render condition
            if episode % render_every == 0:
                env.render()
            #get action from model or random
            action = np.argmax(trained_model.predict(current_state)[0])
            new_state, reward, done, inf = env.step(action)
            new_state = new_state.reshape(1,-1)
            #update reward
            episode_reward += reward
            
            current_state = new_state
            #if episode is done quit the episode
            if done:
                break
        #track rewards over episodes
        test_rewards.append(episode_reward)
        last_rewards_mean = np.mean(test_rewards)
        print(f"Episode {episode} \t Reward {round(episode_reward)} \t Average Reward {round(last_rewards_mean)} \t Steps {step}")
    print("Testing completed!")
    
def train(training_episodes):
    # setting up parameters
    lr = 0.001
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    #Instantiate Agent
    model = DQNAgent(env, lr, gamma, epsilon, epsilon_decay)
    #Train model
    model.train(training_episodes)
def test(filename, test_episodes, render_every):
    trained_model = load_model(filename)
    test_trained_model(env,trained_model,test_episodes, render_every)
    
env = gym.make('LunarLander-v2')

