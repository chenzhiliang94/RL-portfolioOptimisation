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
RISK_FREE_RATE = 1.0008 # daily risk free rate for keeping money as cash instead of investing
MAX_ACCOUNT_BALANCE = 100000
MAX_NUM_SHARES = 1000000
MAX_SHARE_PRICE = 10000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

test_in_sample = False
TRAIN_STEPS = 20000
deterministic_prediction = False
STOCKS = ["DB", "T"]
# divide to training and testing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler(feature_range=(0.25,0.75))


def combine_stock_price(stock_names):
    # join stock data into one df (increase number of cols)
    list_of_stock_data = []
    for stock in stock_names:
        df = pd.read_csv('stocks/' + stock + '.csv')
        df = df.sort_values('Date')
        df.drop(columns=['Date'], inplace=True)
        adj_price_actual = df['Adj Close']
        col_names = df.columns
        df = s.fit_transform(df)
        df = pd.DataFrame(df, columns=col_names)
        df['Adj Close Actual'] = adj_price_actual
        df_col_names = [x+stock for x in df.columns] # rename df columns to include stock name at end
        df.columns = df_col_names
        print(df)
        list_of_stock_data.append(df)
    all_stock_data = pd.concat(list_of_stock_data, axis=1)
    return all_stock_data

df = combine_stock_price(STOCKS)
df_train = df.iloc[:TRAINING_DURATION]
df_test = df.iloc[TRAINING_DURATION:]
df_test.reset_index(inplace=True)
print(df_train.head())

class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # ratio of money put into stock
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float16)
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
        self.action = "BUY"
        self.stocks = []


        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        self.current_step = NUM_DAYS_BACK # don't start from day 0, since we can't observe before that in the data
        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'OpenT'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'HighT'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'LowT'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Adj CloseT'].values ,
            self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'VolumeT'].values ,])
        # Append additional data and scale each value to between 0-1
        print(self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'OpenT'].values)
        print(frame)
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        exit()
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
        reward = ((self.reward * MAX_ACCOUNT_BALANCE) - INITIAL_ACCOUNT_BALANCE) / INITIAL_ACCOUNT_BALANCE

        # done variable is a boolean indicator which terminates the RL.
        # We do not want this to happen; done is True when we become bankrupt
        done = self.net_worth <= 0
        obs = self._next_observation()

        # must returns these 4 items
        return obs, reward, done, {"action":self.action}

    def _take_action(self, action):
        # take_action is a function which changes the observable environment when
        # an action is taken; in our case, an action changes our portfolio.
        # Just some if-else arithmetic to change our balance, shares etc for certain actions

        #current_price = random.uniform(
        #    self.df.loc[self.current_step, "Open"],
        #   self.df.loc[self.current_step, "Adj Close"])

        # if sum of money allocation > 1:
        # it is an invalid action. we penalise. Hopefully, the agent learns not to allocate
        # the portfolio as [0.75, 0.75] for example
        if np.sum(action) > 1:
            self.reward = 0
            self.action = dict.fromkeys(self.action, "DONT DO ANYTHING")
            return

        new_investment_worth = 0 # this will be the worth on next day's closing
        for index, stock_name in enumerate(self.stock):
            yesterday_closing = self.df.loc[self.current_step-1, "Adj Close Actual" + stock_name] # yesterday's closing price
            today_closing = self.df.loc[self.current_step, "Adj Close Actual" + stock_name] # today's closing price. This is how much our
                                                                          # held stock will change in price
            yesterday_shares_held = self.shares_held[stock_name] # shares held at yesterday's closing
            action_type = action[index] # a real value from 0-1
            to_invest = self.net_worth * action_type

            self.shares_held[stock_name] = to_invest / yesterday_closing # we buy the stocks at yesterday's closing price
            self.cost_basis[stock_name] = np.abs(today_closing - yesterday_closing)/yesterday_closing # absolute difference change in stock price
            if yesterday_shares_held > self.shares_held: # we sold
                self.total_shares_sold[stock_name] += (yesterday_shares_held-self.shares_held)
                self.total_sales_value[stock_name] += (yesterday_closing * (yesterday_shares_held-self.shares_held))
                self.action[stock_name] = "SELL"
            elif yesterday_shares_held == self.shares_held:
                self.action[stock_name] = "DONT DO ANYTHING"
            else:
                self.action[stock_name] = "BUY"

            new_investment_worth += self.shares_held[stock_name] * today_closing # new investment worth of this stock
            '''
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
        '''
        to_hold = self.net_worth * (1 - np.sum(self.action)) # whatever's not used to ivnest goes into risk-free balance
        self.balance = to_hold
        self.balance *= RISK_FREE_RATE  # risk free rate for cash at hand
        self.net_worth = self.balance + new_investment_worth # shares held is multipled by today's closing
        self.reward = self.net_worth

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        #if self.shares_held == 0:
        #    self.cost_basis = 0

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
    obs, rewards, done, what_did_we_do = env_test.step(action)
    price.append(obs[0][3][-1])
    print(what_did_we_do)
    strategy.append(what_did_we_do[0]["action"])

    nw,sh = env_test.render()
    net_worth.append(nw)
    shares_held.append(sh)
    '''
    if action[0][0]>=1 and action[0][0]<2 and action[0][1]>0:
      print("sell amount: " + str(action[0][1]))
    elif action[0][0]<1 and action[0][1]>0:
      print("buy amount: " + str(action[0][1]))
    '''

import matplotlib.pyplot as plt
#df_test = pd.DataFrame(s.inverse_transform(df_test), columns=df_test.columns)

buy_action_y = []
buy_action_x = []
sell_action_y = []
sell_action_x = []
for i in range(len(strategy)):
    strat = strategy[i]
    if strat == "BUY": # buy
        buy_action_y.append(price[i])
        buy_action_x.append(range(NUM_DAYS_BACK+1,NUM_DAYS_TEST+NUM_DAYS_BACK+1)[i])
    elif strat=="SELL": # sell
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
