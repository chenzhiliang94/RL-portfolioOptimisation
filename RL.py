import gym
from gym import spaces
import math
import random
import numpy as np

# Since the updated tf is version 2, we have to disable some v2 features to make this work
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# some existing rl models
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan
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


# some parameters; self explanatory
TRAINING_DURATION = 900
NUM_DAYS_BACK = 21 # how many past days observations to take note of before making a decision
FEATURES = 5 # how many features to take note per day
NUM_DAYS_TEST = 300 # how many testing days to plot
RISK_FREE_RATE = 1.00008 # daily risk free rate for keeping money as cash instead of investing
MAX_ACCOUNT_BALANCE = 30000
MAX_NUM_SHARES = 10000
MAX_SHARE_PRICE = 10000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000 # initial money to have
PLOT_INTO_FUTURE = 30 # plot the prices into the future after last trade
test_in_sample = False # whether to run test on in sample data or out of sample
TRAIN_STEPS = 2500 # how many iterations to train
INVALID_PENALTY = -10 # unused currently
gamma = 0.99 # delay factor of reward
LEARNING_RATE = 0.001
ENTROPY_COEFFICIENT = 0.75 # how much randomness to add in during training to encourage exploration
STEPS = 128 # how many samples to use per iteration
UPPER_BOUND_INVESTMENT = 1 # unused
deterministic_prediction = True
to_plot_stock = "GOOG" # when we plot our networth throughout time, we also plot this stock's price movement
STOCKS = [ "DB","GOOG", "NFLX", "GS", "MCD", "WFC", "BAC", "T"] # the stocks; the csv file must be present in direction /stocks

# reward function; simply the immediate reward we get from one trade (change in yesterday's networth with today's)
# notice the reward is divided by a constant to keep reward value small
def reward(yesterday_net_worth, today_net_worth):
    return ((today_net_worth)-(yesterday_net_worth))/INITIAL_ACCOUNT_BALANCE

# divide to training and testing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
s = MinMaxScaler(feature_range=(0.1,0.9))

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

class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock):
        super(StockTradingEnvironment, self).__init__()
        self.stock = stock
        self.df = df
        self.shares_held = {key: 0 for key in stock}
        self.cost_basis = {key: 0 for key in stock}
        self.total_shares_sold = {key: 0 for key in stock}
        self.total_sales_value = {key: 0 for key in stock}
        self.action = {key: "DONT DO ANYTHING" for key in stock}

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # ratio of money put into stock
        self.action_space = spaces.Box(
            low=np.array([0.] * len(STOCKS)), high=np.array([UPPER_BOUND_INVESTMENT] * len(STOCKS)), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1, (NUM_DAYS_BACK+1) * FEATURES * len(self.stock) + 4 * len(self.stock) + 2 ), dtype=np.float16) # +1 because we include own observation

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.prev_day_net_worth = INITIAL_ACCOUNT_BALANCE
        self.reward = 0
        self.stock = STOCKS
        self.shares_held = {key: 0 for key in self.stock}
        self.cost_basis = {key: 0 for key in self.stock}
        self.total_shares_sold = {key: 0 for key in self.stock}
        self.total_sales_value = {key: 0 for key in self.stock}
        self.action = {key: "DONT DO ANYTHING" for key in self.stock}


        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        self.current_step = NUM_DAYS_BACK # don't start from day 0, since we can't observe before that in the data
        return self._next_observation()

    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([])
        for stock_name in self.stock:

            temp = [
                self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Open'+stock_name].values ,
                self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'High'+stock_name].values ,
                self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Low'+stock_name].values ,
                self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Adj Close'+stock_name].values ,
                self.df.loc[self.current_step-NUM_DAYS_BACK: self.current_step , 'Volume'+stock_name].values ,]
            for v in temp:
                frame = np.append(frame, v)
        print("cost basis")
        print(np.array(list(self.cost_basis.values())))
        frame = np.append(frame, 0.001+ np.array(list(self.shares_held.values()))/MAX_NUM_SHARES) # append shares held as obs
        frame = np.append(frame, 0.001+np.array(list(self.cost_basis.values()))) # append cost basis held as obs
        frame = np.append(frame, 0.001+np.array(list(self.total_shares_sold.values()))/ 100000)  # append shares sold as obs
        frame = np.append(frame, 0.001+np.array(list(self.total_sales_value.values()))/ (MAX_NUM_SHARES * MAX_SHARE_PRICE))  # append shares sales value as obs
        frame = np.append(frame, 0.001+np.array([self.balance/ MAX_ACCOUNT_BALANCE])) # one element
        frame = np.append(frame, 0.001+np.array([self.max_net_worth/ MAX_ACCOUNT_BALANCE])) # one element
        return frame
        '''
        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)
        return obs
        '''

    def step(self, action):
        self.prev_day_net_worth = self.net_worth
        # Execute one time step within the environment
        self.current_step += 1
        # take_action calculates the new balance due to the action taken (with the new stock price)
        self._take_action(action)

        # if we have reached last row of training data
        # reset to the start of training data
        if self.current_step == len(self.df.loc[:, 'OpenT'].values)-1:
            self.current_step = NUM_DAYS_BACK
            self.reset()

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
        return obs, reward, done, {"action":self.action["T"]}

    def _take_action(self, action):
        # take_action is a function which changes the observable environment when
        # an action is taken; in our case, an action changes our portfolio.
        # Just some if-else arithmetic to change our balance, shares etc for certain actions

        # if sum of money allocation > 1:
        # it is an invalid action. we penalise. Hopefully, the agent learns not to allocate
        # the portfolio as [0.75, 0.75] for example
        norm = np.sum(action)
        if not norm == 0: # normalise action (if not all zeroes)
            action = [a / norm for a in action]

        if np.sum(action) > 1:
            print("invalid action during training")
            self.reward = INVALID_PENALTY
            self.action = dict.fromkeys(self.action, "DONT DO ANYTHING")
            return

        new_investment_worth = 0 # this will be the worth on next day's closing
        for index, stock_name in enumerate(self.stock):
            print("stock name: " + stock_name)
            yesterday_closing = self.df.loc[self.current_step-1, "Adj Close Actual" + stock_name] # yesterday's closing price
            today_closing = self.df.loc[self.current_step, "Adj Close Actual" + stock_name] # today's closing price. This is how much our
                                                                          # held stock will change in price
            yesterday_shares_held = self.shares_held[stock_name] # shares held at yesterday's closing
            action_type = action[index] # a real value from 0-1
            to_invest = self.net_worth * action_type
            print("yesterday's closing: " + str(yesterday_closing))
            print("today's closing: " + str(today_closing))
            print("yesterday share amount held: " + str(yesterday_shares_held))
            print("yesterday's networth: " + str(self.net_worth))
            print("percentage networth put into this stock: " + str(action[index]))
            if (math.isnan(action[index])):
                print(action)
                exit()
            self.shares_held[stock_name] = to_invest / yesterday_closing # we buy the stocks at yesterday's closing price
            self.cost_basis[stock_name] = np.abs(today_closing - yesterday_closing)/yesterday_closing # absolute difference change in stock price
            if yesterday_shares_held > self.shares_held[stock_name]: # we sold
                self.total_shares_sold[stock_name] += (yesterday_shares_held-self.shares_held[stock_name])
                self.total_sales_value[stock_name] += (yesterday_closing * (yesterday_shares_held-self.shares_held[stock_name]))
                self.action[stock_name] = "SELL"
            elif yesterday_shares_held == self.shares_held[stock_name]:
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
        print("actions")
        print(action)
        to_hold = self.net_worth * (1 - np.sum(action)) # whatever's not used to invest goes into risk-free balance
        print("today's  available cash:")
        print(to_hold)
        print("today's investment worth:")
        print(new_investment_worth)
        self.balance = to_hold
        self.balance *= RISK_FREE_RATE  # risk free rate for cash at hand
        self.net_worth = self.balance + new_investment_worth # shares held is multipled by today's closing
        self.reward = reward(self.prev_day_net_worth, self.net_worth) # reward is percentage change of networth
        print("today's net worth:")
        print(self.net_worth)
        print("immediate reward:")
        print(self.reward)
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


env = DummyVecEnv([lambda: StockTradingEnvironment(df_train, STOCKS)])
env = VecCheckNan(env, raise_exception=True)
nb_actions = env.action_space.shape[0]

print("learning...")
# proximal policy optimisation, these two lines of code trains the MlpPolicy using PPO2 algorithm
model = PPO2(MlpPolicy, env, verbose=1,gamma=gamma,learning_rate=LEARNING_RATE, seed=10, n_steps=STEPS, cliprange=0.2,
             ent_coef=ENTROPY_COEFFICIENT)
model.learn(total_timesteps=TRAIN_STEPS)

if (test_in_sample):
    df_test=df_train
env_test = DummyVecEnv([lambda: StockTradingEnvironment(df_test, STOCKS)])
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
    index = (STOCKS.index(to_plot_stock) * (NUM_DAYS_BACK + 1) * FEATURES + ((NUM_DAYS_BACK+1)*4-1)) # 4th column is adj close
    price.append(obs[0][0][index])
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
plt.title("Strategy by RL on " + to_plot_stock +" stocks")
plt.plot(df_test['Adj Close'+to_plot_stock].iloc[:NUM_DAYS_TEST+NUM_DAYS_BACK+2+PLOT_INTO_FUTURE], label='stock closing price',c='black') # plot actual movement of stock
plt.scatter(buy_action_x, buy_action_y,c='green', label="buy",s=10) # plot all buy actions
plt.scatter(sell_action_x, sell_action_y,c='red', label="sell",s=10) # plot all sell actions
plt.scatter(NUM_DAYS_BACK+1, df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_BACK+1],marker='x',c='black') # plot day 1 price
plt.scatter(NUM_DAYS_TEST+NUM_DAYS_BACK+1, df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1],marker='x',c='black') # plot last day end price
plt.annotate('start: '+str(round(df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_BACK+1],2)),(NUM_DAYS_BACK+1, 0.9*df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_BACK+1]))
plt.annotate('end: '+str(round(df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1],2)),(NUM_DAYS_TEST+NUM_DAYS_BACK+1, 0.9*df_test['Adj Close'+to_plot_stock].iloc[NUM_DAYS_TEST+NUM_DAYS_BACK+1]))
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
