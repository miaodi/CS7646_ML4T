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

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dmiao3"  # replace tb34 with your Georgia Tech username.


def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 900897987  # replace with your GT ID number


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
    print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments


# if __name__ == "__main__":
#     test_code()

def indexGen(size):
    res = []
    for i in range(size):
        res.append("Episode-"+str(i+1))
    return res


def gamble(num_of_episodes, min_winning=80, rate=.5, num_of_runs=1000, max_lose=-1000000000):
    res = np.full((num_of_runs+1, num_of_episodes), max_lose)
    res[0, :] = 0
    for i in range(num_of_episodes):
        bet_amount = 1
        won = False
        for run in range(1, num_of_runs+1):
            bet_amount = min(bet_amount, res[run-1, i]-max_lose)
            if bet_amount == 0:
                break
            if res[run-1, i] >= min_winning:
                res[run:, i] = min_winning
                break
            won = get_spin_result(rate)
            if won == True:
                res[run, i] = res[run-1, i]+bet_amount
                bet_amount = 1
            else:
                res[run, i] = res[run-1, i]-bet_amount
                bet_amount *= 2

    df = pd.DataFrame(data=res, columns=indexGen(num_of_episodes))
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['mean+std'] = df['mean'] + df['std']
    df['mean-std'] = df['mean'] - df['std']
    df['median'] = df.median(axis=1)
    df['median+std'] = df['median'] + df['std']
    df['median-std'] = df['median'] - df['std']
    return df

# 3.2 Experiment 1 – Explore the strategy and create some charts

df = gamble(1000, max_lose=-1000000000000, rate=18./38)
x_bounds = [0, 300]
y_bounds = [-256, 100]
ax = df.iloc[:, 0:10].plot()
ax.set_xlabel("Spin")
ax.set_ylabel("Winnings")

ax.set_xlim([x_bounds[0], x_bounds[1]])
ax.set_ylim([y_bounds[0], y_bounds[1]])
plt.savefig("Exp1_Fig1")

ax = df[['mean', 'mean+std', 'mean-std']].plot()
ax.set_xlabel("Spin")
ax.set_ylabel("Winnings")

ax.set_xlim([x_bounds[0], x_bounds[1]])
ax.set_ylim([y_bounds[0], y_bounds[1]])
plt.savefig("Exp1_Fig2")


ax = df[['median', 'median+std', 'median-std']].plot()
ax.set_xlabel("Spin")
ax.set_ylabel("Winnings")

ax.set_xlim([x_bounds[0], x_bounds[1]])
ax.set_ylim([y_bounds[0], y_bounds[1]])
plt.savefig("Exp1_Fig3")

# 3.3 Experiment 2 – A more realistic gambling simulator  

df_256 = gamble(1000, max_lose=-256, rate=18./38)

ax = df_256[['mean', 'mean+std', 'mean-std']].plot()
ax.set_xlabel("Spin")
ax.set_ylabel("Winnings")

ax.set_xlim([x_bounds[0], x_bounds[1]])
ax.set_ylim([y_bounds[0], y_bounds[1]])
plt.savefig("Exp2_Fig4")


ax = df_256[['median', 'median+std', 'median-std']].plot()
ax.set_xlabel("Spin")
ax.set_ylabel("Winnings")

ax.set_xlim([x_bounds[0], x_bounds[1]])
ax.set_ylim([y_bounds[0], y_bounds[1]])
plt.savefig("Exp2_Fig5")
