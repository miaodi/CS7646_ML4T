""""""
"""Assess a betting strategy.  		  	   		   	 		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  

Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  

-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  

Student Name: Chengqi Huang (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: Chuang405 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903534690 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""


import numpy as np
import matplotlib.pyplot as plt


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xxxxx"  # replace tb34 with your Georgia Tech username.


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 12341234  # replace with your GT ID number


def get_spin_result(win_prob):
    """
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.

    :param win_prob: The probability of winning
    :type win_prob: float
    :return: The result of the spin.
    :rtype: bool
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def test_code():
    """
    Method to test your code
    """
    win_prob = 0.60  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments


def episode():
    episode_winnings = 0
    times = 0
    win_prob = 9 / 19
    result = np.ones(1001, dtype=np.int_) * 80
    result

    result[0] = 0
    while (episode_winnings < 80) and (times < 1000):
        won = False
        bet_amount = 1

        while (not won) and (times < 1000):
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bet_amount = bet_amount * 2

            times = times + 1
            result[times] = episode_winnings

            # print(won)
            # print(episode_winnings)
            # print(times)
    # print(result)
    return result




def plot_1():
    for i in range(0, 10):
        simulation_episode = episode()
        plt.plot(simulation_episode)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Times")
    plt.ylabel('Winnings')
    plt.title('Figure 1')
    plt.legend(labels=['Episode1', 'Episode2', 'Episode3', 'Episode4', 'Episode5', 'Episode6', 'Episode7', 'Episode8',
                       'Episode9', 'Episode10'], loc='best')
    plt.savefig('Figure1.png')

    # plt.show

    plt.close()

    return 0




def bets_1000():
    winnings = []
    for i in range(0,1000):
        winnings.append(episode())
    np_winnings = np.array(winnings)
    return np_winnings


def plot_2(a):
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    upper = mean + std
    lower = mean - std

    # print(mean[19])
    # print(std[19])
    # print(upper[19])
    # print(mean[19] + std[19])
    plt.plot(mean)
    plt.plot(upper)
    plt.plot(lower)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Times")
    plt.ylabel('Winnings')
    plt.title('Figure 2')
    plt.legend(labels=['mean', 'mean + std', 'mean - std'], loc='best')
    # plt.show
    plt.savefig('Figure2.png')
    plt.close()

    return 0


def plot_3(a):
    median = np.median(a, axis=0)

    std = a.std(axis=0)
    upper = median + std
    lower = median - std

    # print(mean[19])
    # print(std[19])
    # print(upper[19])
    # print(mean[19] + std[19])
    plt.plot(median)
    plt.plot(upper)
    plt.plot(lower)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Times")
    plt.ylabel('Winnings')
    plt.title('Figure 3')
    plt.legend(labels=['median', 'median + std', 'median - std'], loc='best')
    # plt.show
    plt.savefig('Figure3.png')
    plt.close()
    return 0


def episode_2():
    episode_winnings = 0
    times = 0
    win_prob = 9 / 19
    result = np.ones(1001, dtype=np.int_) * 80
    bankroll = 256

    result[0] = 0
    loose = 0

    while (episode_winnings < 80) and (times < 1000) and (bankroll >= 0):
        won = False
        bet_amount = 1

        while (not won) and (times < 1000) and (bankroll >= 0):
            # wager bet_amount on black
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet_amount
                bankroll = bankroll + bet_amount
            else:
                episode_winnings = episode_winnings - bet_amount
                bankroll = bankroll - bet_amount
                bet_amount = bet_amount * 2

            if bankroll <= 0:
                times = times + 1
                result[times:] = -256
                loose = 1
                break
            if bankroll < bet_amount:
                bet_amount = bankroll

            times = times + 1
            result[times] = episode_winnings

            # print('win or loose',won)
            # print('episode_winnings', episode_winnings)
            # print('how many times',times)
            # print('bankroll', bankroll)
    # print(result)
    return result, loose

def bets_1000_realistic():
    winnings = []
    loose = 0
    for i in range(0,1000):
        a,b = episode_2()
        loose += b
        winnings.append(a)
    np_winnings = np.array(winnings)
    return np_winnings, loose


def plot_4(a):
    mean = a.mean(axis=0)
    std = a.std(axis=0)
    upper = mean + std
    lower = mean - std

    # print(mean[19])
    # print(std[19])
    # print(upper[19])
    # print(mean[19] + std[19])
    plt.plot(mean)
    plt.plot(upper)
    plt.plot(lower)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Times")
    plt.ylabel('Winnings')
    plt.title('Figure 4')
    plt.legend(labels=['mean', 'mean + std', 'mean - std'], loc='best')
    # plt.show
    plt.savefig('Figure4.png')
    plt.close()

    return 0


def plot_5(a):
    median = np.median(a, axis=0)

    std = a.std(axis=0)
    upper = median + std
    lower = median - std

    # print(mean[19])
    # print(std[19])
    # print(upper[19])
    # print(mean[19] + std[19])
    plt.plot(median)
    plt.plot(upper)
    plt.plot(lower)
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Times")
    plt.ylabel('Winnings')
    plt.title('Figure 5')
    plt.legend(labels=['median', 'median + std', 'median - std'], loc='best')
    # plt.show
    plt.savefig('Figure5.png')
    plt.close()
    return 0

if __name__ == "__main__":
    # test_code()
    episode()
    plot_1()
    np_winnings = bets_1000()
    plot_2(np_winnings)
    plot_3(np_winnings)
    episode_2()
    np_winnings_realistic, loose = bets_1000_realistic()
    plot_4(np_winnings_realistic)
    plot_5(np_winnings_realistic)



