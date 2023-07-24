import ManualStrategy
import os

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'plots/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)
ManualStrategy.optimize()
ManualStrategy.InSample()
ManualStrategy.OutOfSample()