import gym
from gym import spaces
import random
import numpy as np
import json
import datetime as dt
from rl.agents.ddpg import DDPGAgent
from rl.agents.dqn import DQNAgent
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# some existing rl models
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, DQN, SAC, A2C

# actor critic model; needs to take in custom model for actor and critic
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

'''
# deep learning modules for actor/critic model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
'''
tf.disable_v2_behavior()

TRAINING_DURATION = 900
NUM_DAYS_BACK = 5 # how many past days observations to take note of before making a decision
FEATURES = 6 # how many features to take note per day
NUM_DAYS_TEST = 300 # how many testing days to plot

MAX_ACCOUNT_BALANCE = 100000
MAX_NUM_SHARES = 1000000
MAX_SHARE_PRICE = 10000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

test_in_sample = False
TRAIN_STEPS = 5000
deterministic_prediction = False


class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(FEATURES, NUM_DAYS_BACK+1), dtype=np.float16) # +1 because we include own observation

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.reward = 0

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        self.current_step = NUM_DAYS_BACK # don't start from day 0, since we can't observe before that in the data
        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Open'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'High'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Low'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Adj Close'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Volume'].values ,])
        # Append additional data and scale each value to between 0-1

        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1
        # take_action calculates the new balance due to the action taken (with the new stock price)
        self._take_action(action)

        # if we have reached last row of training data
        # reset to the start of training data
        if self.current_step == len(self.df.loc[:, 'Open'].values)-1:
            self.current_step = NUM_DAYS_BACK

        # reward function design - play with this!
        # net worth here denotes the net worth in the NEW DAY (the first line
        # has already increased the step by 1)
        delay_modifier_a = ((MAX_STEPS-self.current_step) / MAX_STEPS)
        delay_modifier_b = (self.current_step) / MAX_STEPS
        reward = self.reward

        # done variable is a boolean indicator which terminates the RL.
        # We do not want this to happen; done is True when we become bankrupt
        done = self.net_worth <= 0
        obs = self._next_observation()

        # must returns these 4 items
        return obs, reward, done, {}

    def _take_action(self, action):
        # take_action is a function which changes the observable environment when
        # an action is taken; in our case, an action changes our portfolio.
        # Just some if-else arithmetic to change our balance, shares etc for certain actions

        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"],
            self.df.loc[self.current_step, "Adj Close"])
        self.balance *= 1.00008
        current_price = self.df.loc[self.current_step, "Adj Close"]
        action_type = action[0]
        amount = action[1]
        if action_type < 1:
            # Buy amount % of balance in shares
            if self.balance == 0:
                # we penalise if the agent attempts to buy when we have no balance!
                self.reward = 0
                return
            total_possible = self.balance / current_price
            shares_bought = total_possible * amount
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            if self.shares_held == 0:
            #    # we penalise if the agent attempts to sell when we have no balance!
                self.reward = 0
                return
            shares_sold = self.shares_held * amount
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        # balance (whatever cash we have at hand) is multiplied by a risk free rate
        # this is daily

        self.net_worth = self.balance + self.shares_held * current_price
        self.reward = self.net_worth

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return self.net_worth, self.shares_held

# divide to training and testing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler(feature_range=(0.25,0.75))
df = pd.read_csv('stocks/T.csv')
df = df.sort_values('Date')
df.drop(columns=['Date'],inplace=True)
col_names = df.columns
df = s.fit_transform(df)

df = pd.DataFrame(df, columns=col_names)

df_train = df.iloc[:TRAINING_DURATION]
df_test = df.iloc[TRAINING_DURATION:]
df_test.reset_index(inplace=True)

env = DummyVecEnv([lambda: StockTradingEnvironment(df_train)])
nb_actions = env.action_space.shape[0]

print("learning...")
# proximal policy optimisation, these two lines of code trains the MlpPolicy using PPO2 algorithm
model = PPO2(MlpPolicy, env, verbose=1,gamma=0.95,learning_rate=0.001)
model.learn(total_timesteps=TRAIN_STEPS)

if (test_in_sample):
    df_test=df_train
env_test = DummyVecEnv([lambda: StockTradingEnvironment(df_test)])
obs = env_test.reset()


print("testing...")
strategy = []
price = []
net_worth = []
shares_held = []

# printing stuff
# day 5 - 305
for i in range(NUM_DAYS_TEST):
    action, _states = model.predict(obs,deterministic=deterministic_prediction)
    strategy.append(action)
    obs, rewards, done, info = env_test.step(action)
    price.append(obs[0][3][-1])

    nw,sh = env_test.render()
    net_worth.append(nw)
    shares_held.append(sh)
    if action[0][0]>=1 and action[0][0]<2 and action[0][1]>0:
      print("sell amount: " + str(action[0][1]))
    elif action[0][0]<1 and action[0][1]>0:
      print("buy amount: " + str(action[0][1]))

import matplotlib.pyplot as plt
#df_test = pd.DataFrame(s.inverse_transform(df_test), columns=df_test.columns)
buy_action_y = []
buy_action_x = []
sell_action_y = []
sell_action_x = []
for i in range(len(strategy)):
    strat = strategy[i]
    if strat[0][0]<1 and strat[0][1]>0: # buy
        buy_action_y.append(price[i])
        buy_action_x.append(range(NUM_DAYS_BACK+1,NUM_DAYS_TEST+NUM_DAYS_BACK+1)[i])
    elif strat[0][0]>=1 and strat[0][0]<2 and strat[0][1]>0: # sell
        sell_action_y.append(price[i])
        sell_action_x.append(range(NUM_DAYS_BACK+1,NUM_DAYS_TEST+NUM_DAYS_BACK+1)[i])

# actions are taken in day 5.5 to 306.5 (at night when stock closes)
plt.subplot(211)
plt.title("Strategy by RL on T stocks")
plt.plot(df_test['Adj Close'].iloc[:NUM_DAYS_TEST+NUM_DAYS_BACK+2], label='stock closing price',c='black') # plot actual movement of stock
plt.scatter(buy_action_x, buy_action_y,c='green', label="buy",s=10) # plot all buy actions
plt.scatter(sell_action_x, sell_action_y,c='red', label="sell",s=10) # plot all sell actions
plt.scatter(NUM_DAYS_BACK+1, df_test['Adj Close'].iloc[NUM_DAYS_BACK+1],marker='x',c='black') # plot day 1 price
plt.scatter(NUM_DAYS_TEST+NUM_DAYS_BACK+1, df_test['Adj Close'].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1],marker='x',c='black') # plot last day end price
plt.annotate('start: '+str(round(df_test['Adj Close'].iloc[NUM_DAYS_BACK+1],2)),(NUM_DAYS_BACK+1, 0.9*df_test['Adj Close'].iloc[NUM_DAYS_BACK+1]))
plt.annotate('end: '+str(round(df_test['Adj Close'].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1],2)),(NUM_DAYS_TEST+NUM_DAYS_BACK+1, 0.9*df_test['Adj Close'].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1]))
plt.legend()

plt.subplot(212)
plt.plot(range(NUM_DAYS_BACK+1, NUM_DAYS_TEST+NUM_DAYS_BACK+1), net_worth,label='net worth') # plot net worth throughout time
plt.scatter(NUM_DAYS_BACK+1, net_worth[0],marker='x',c='black') # plot day 1 net worth (not starting day)
plt.scatter(NUM_DAYS_TEST+NUM_DAYS_BACK+1, net_worth[-1],marker='x',c='black') # plot end net worth
plt.annotate('start: '+str(round(net_worth[0],0)),(NUM_DAYS_BACK+1, net_worth[0]-100))
plt.annotate('end: '+str(round(net_worth[-1],0)),(NUM_DAYS_TEST+NUM_DAYS_BACK+1, net_worth[-1]-100))
plt.legend()
plt.show()
plt.clf()
