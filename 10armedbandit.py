import numpy as np
import matplotlib.pyplot as plt
import time

# Bandit class containing the states and actions, and the overall rules of the test
class Bandit(object):

    
    def __init__(self, nArms, mean, stDev):

        # Number of arms
        self.no_Arms = nArms

        # meand and standard deviation for probability distribution
        self.mean = mean        
        self.STD = stDev      

        self.actionVal = np.zeros(nArms)   # Array to store action values
        self.optimal = 0                  # Store optimal value for greedy
        self.reset()

    # Reset Bandit for next iteration
    def reset(self):
        # Setting random values for each action
        self.actionVal = np.random.normal(self.mean, self.STD, self.no_Arms)

        # maximum value in action value
        self.optimal = np.argmax(self.actionVal)



# User Class - Controls the agents movement and behaviour in the Env interacting with the Bandit
# and receives information on the current position
class User(object):

    # Constructor
    def __init__(self,nArms, eProb=0):
        self.no_Arms = nArms      
        self.eProb = eProb     

        self.t_step = 0                    
        self.lastAct = None               # Store last action

        self.countAction = np.zeros(nArms)          # count of actions taken at time t
        self.rewardSum = np.zeros(nArms)             # Sums number of rewards
        self.actionValEsti = np.zeros(nArms)     # action value estimates sum(rewards)/Amount


    # Return string for graph legend
    def __str__(self):
        if self.eProb == 0:
            return "Greedy"
        else:
            return "Epsilon = " + str(self.eProb)


    # Selects action based on a epsilon-greedy behaviour,
   
    def action(self):
        
        # Epsilon method
        randProb = np.random.random()   # Pick random probability between 0-1
        # If, probability less than e, pick random action
        if randProb < self.eProb:
            a = np.random.choice(len(self.actionValEsti))    

        # Greedy Method
        else:
            maxAction_Val = np.argmax(self.actionValEsti)     
            # identify the corresponding action, as array containing only actions with max
            action = np.where(self.actionValEsti == np.argmax(self.actionValEsti))[0]

            # If multiple actions contain the same value, randomly select an action
            if len(action) == 0:
                a = maxAction_Val
            else:
                a = np.random.choice(action)

        # save last action in variable, and return result
        self.lastAct = a
        return a


    # Update class - updates the value extimates amounts based on the last action
    def Update(self, reward):
        # Add 1 to the number of action taken in step
        La = self.lastAct

        self.countAction[La] += 1       # Add 1 to action selection
        self.rewardSum[La] += reward     # Add reward to sum array

        # Calculate new action-value, sum(r)/ka
        self.actionValEsti[La] = self.rewardSum[La]/self.countAction[La]

        self.t_step += 1


    # Reset all variables for next iteration
    def reset(self):
        self.t_step = 0                    # Time Step t
        self.lastAct = None              

        self.countAction[:] = 0                 
        self.rewardSum[:] = 0
        self.actionValEsti[:] = 0   # action value estimates Qt ~= Q*(a)



# Env class to control all objects (User/Bandit)
class Env(object):

    # Constructor
    def __init__(self, Bandit, Users, plays, iterations):
        self.Bandit = Bandit
        self.Users = Users

        self.plays = plays
        self.iter = iterations


    # Run Test
    def play(self):

        # Array to store the scores, number of plays X number of Users
        score_Array = np.zeros((self.plays, len(self.Users)))
        # Optimal Action values
        optimal_Array = np.zeros((self.plays, len(self.Users)))

       # Loop through each user 
        for i in range(self.iter):

            #Reset Bandit and all Users
            self.Bandit.reset()
            for User in self.Users:
                User.reset()

            # Play for each user    
            for play in range(self.plays):
                agent_count = 0

                for each_user in self.Users:
                    act_taken =  each_user.action()

                    # Reward for action taken
                    rewardT = np.random.normal(self.Bandit.actionVal[act_taken], scale=1)

                    # Update agent
                    each_user.Update(reward=rewardT)

                    # Add score in arrary, graph 1
                    score_Array[play,agent_count] += rewardT

                    # check the optimal action, add optimal to array, graph 2
                    if act_taken == self.Bandit.optim:
                        optimal_Array[play,agent_count] += 1

                    agent_count += 1

        #return averages
        scoreAvg = score_Array/self.iter
        optimlAvg = optimal_Array/self.iter

        return scoreAvg, optimlAvg




if __name__ == "__main__":
    start_time = time.time()    
    nArms = 10                  
    iterations = 2000         
    plays = 1000             

    # Bandit parameters, mean and standard deviation
    bandit = Bandit(nArms=nArms,mean=0,stDev=1)
    # User parameters, epsilon, number of users
    Users = [User(nArms=nArms),User(nArms=nArms,eProb=0.1),User(nArms=nArms,eProb=0.01)]
    # Create Env
    env = Env(Bandit=Bandit,Users=Users,plays=plays,iterations=iterations)

    # Run Env
    print("Running...")
    avg_reward, optimal_reward = Env.play()
    


    #Graph 1 - Averate rewards over all plays
    plt.title("10-Armed Bandit - Average Rewards")
    plt.plot(avg_reward)
    plt.ylabel('Average Reward')
    plt.xlabel('Plays')
    plt.legend(Users, loc=4)
    plt.show()

    #Graph 1 - optimal selections over all plays
    plt.title("10-Armed Bandit - % Optimal Action")
    plt.plot(optimal_reward * 100)
    plt.ylim(0, 100)
    plt.ylabel('% Optimal Action')
    plt.xlabel('Plays')
    plt.legend(Users, loc=4)
    plt.show()