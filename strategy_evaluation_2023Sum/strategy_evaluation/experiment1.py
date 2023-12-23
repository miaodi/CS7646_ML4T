
import os
import StrategyLearner

def author():
    return ('dmiao3')

def exec():
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
        
    StrategyLearner.InSample()
    StrategyLearner.OutOfSample()


if __name__ == "__main__":
    exec()