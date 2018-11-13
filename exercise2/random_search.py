import logging

logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns

from hpbandster.optimizers import RandomSearch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker
import argparse

import cnn_mnist
from cnn_mnist import mnist#, train_and_validate, test

import numpy as np
import json
import os


class MyWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = mnist("./")

    def compute(self, config, budget, **kwargs):
        """
        Evaluates the configuration on the defined budget and returns the validation performance.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train
        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        lr = config["learning_rate"]
        num_filters = config["num_filters"]
        batch_size = config["batch_size"]
        filter_size = config["filter_size"]
        epochs = budget

        print("CONFIG: ", config)
        print("BUDGET: ", budget)
        print("KWARGS: ", kwargs)

        # Train and validate your convolutional neural networks here
        cnn_mnist.num_filters = num_filters
        cnn_mnist.filter_size = filter_size
        cnn_mnist.lr = lr
        learning_curve, model = cnn_mnist.train_and_validate(self.x_train, self.y_train, self.x_valid, self.y_valid, epochs, lr, num_filters, batch_size, filter_size)

        print("learning_curve: ", learning_curve)

        validation_error = 1-float(learning_curve[-1])
        

        return ({
            'loss': validation_error,  # this is the a mandatory field to run hyperband
            'info': {}  # can be used for any user-defined information - also mandatory
        })

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-4, upper=1e-1, default_value=1e-3, log=True)
        
        batch_size = CSH.UniformIntegerHyperparameter('batch_size', lower=16, upper=128, default_value=128, log=True)
        
        num_filters = CSH.UniformIntegerHyperparameter('num_filters', lower=8, upper=64, default_value=16, log=True)
        
        filter_size = CSH.CategoricalHyperparameter('filter_size', [3, 5], default_value=3)

        config_space.add_hyperparameters([learning_rate, batch_size, num_filters, filter_size])

        return config_space


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
# Changed default values for budget from 12 to 6 and for n_iterations from 20 to 50.
parser.add_argument('--budget', type=float,
                    help='Maximum budget used during the optimization, i.e the number of epochs.', default=6)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=50)
args = parser.parse_args()

# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
w = MyWorker(nameserver='127.0.0.1', run_id='example1')
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run RandomSearch, but that is not essential.
# The run method will return the `Result` that contains all runs performed.

rs = RandomSearch(configspace=w.get_configspace(),
                  run_id='example1', nameserver='127.0.0.1',
                  min_budget=int(args.budget), max_budget=int(args.budget))
res = rs.run(n_iterations=args.n_iterations)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
rs.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds information about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])


# Plots the performance of the best found validation error over time
all_runs = res.get_all_runs()

# Could cause problems on the pc pools
# Let's plot the observed losses grouped by budget,
# import hpbandster.visualization as hpvis
# hpvis.losses_over_time(all_runs)
# import matplotlib.pyplot as plt
# plt.savefig("hn_random_search.png")



# Retrain the best configuration (called incumbent) and compute the test error

bestconfig = id2config[incumbent]['config']

# hyperparameters
lr = bestconfig['learning_rate']
num_filters = bestconfig['num_filters']
batch_size = bestconfig['batch_size']
filter_size = bestconfig['filter_size']
epochs = 12

# train and test convolutional neural network
learning_curve, model = cnn_mnist.train_and_validate(w.x_train, w.y_train, w.x_valid, w.y_valid, epochs, lr, num_filters, batch_size, filter_size)

test_error = cnn_mnist.test(w.x_test, w.y_test, model)

# save results in a dictionary and write them into a .json file
results = dict()
results["lr"] = lr
results["num_filters"] = num_filters
results["batch_size"] = batch_size
results["filter_size"] = filter_size
results["learning_curve"] = ["{}".format(i) for i in learning_curve]
results["test_error"] = test_error

# print("learning_curve", learning_curve)

path = os.path.join("./", "results")
os.makedirs(path, exist_ok=True)

fname = os.path.join(path, "hn_results_run_%d.json" % 9999)

fh = open(fname, "w")
json.dump(results, fh)
fh.close()



# Fix for plotting problems on pc pools
# print("all_runs:", all_runs)

all_runs_serialized = {}
for i, v in enumerate(all_runs):
    # print(v)
    # print(type(v))    
    
    all_runs_serialized[i] = {}
    all_runs_serialized[i]['config_id'] = v.config_id
    all_runs_serialized[i]['budget'] = v.budget
    all_runs_serialized[i]['loss'] = v.loss
    all_runs_serialized[i]['time_stamps'] = "{submitted} (submitted), {started} (started), {finished} (finished)\n".format(**v.time_stamps)
    all_runs_serialized[i]['info'] = v.info

fname = os.path.join(path, "hn_allruns_results_run_%d.json" % 9999)

fh = open(fname, "w")
json.dump(all_runs_serialized, fh)
fh.close()

# Compare source of:
#     import hpbandster.visualization as hpvis
#     hpvis.losses_over_time(all_runs)
import matplotlib.pyplot as plt
get_loss_from_run_fn = lambda r: r.loss
budgets = set([r.budget for r in all_runs])
data = {}
for b in budgets:
    data[b] = []
for r in all_runs:
    if r.loss is None:
        continue
    b = r.budget
    t = r.time_stamps['finished']
    l = get_loss_from_run_fn(r)
    data[b].append((t,l))

for b in budgets:
    data[b].sort()


fig, ax = plt.subplots()

for i, b in enumerate(budgets):
    data[b] = np.array(data[b])
    #print(data[b])
    #print(data[b][:,0])
    #print(data[b][:,1])
    #print(range(len(data[b][:,0])))
    ax.scatter(range(1, len(data[b][:,0])+1), data[b][:,1], label='data')
    
    ax.step(range(1, len(data[b][:,0])+1), np.minimum.accumulate(data[b][:,1]), where='post')

ax.set_title('Validation errors over the different iterations')
ax.set_xlabel('iteration')
ax.set_ylabel('validation error')
plt.xlim(0, len(data[b][:,0])+1)
ax.legend()
plt.savefig("hn_random_search.png")
