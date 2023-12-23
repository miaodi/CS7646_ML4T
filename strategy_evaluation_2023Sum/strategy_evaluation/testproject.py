import ManualStrategy
import os
import experiment1
import experiment2

def author():
    return ('dmiao3')

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots/')

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    ManualStrategy.optimize()
    ManualStrategy.InSample()
    ManualStrategy.OutOfSample()

    experiment1.exec()
    experiment2.exec()