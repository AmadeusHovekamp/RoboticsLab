import json
import matplotlib.pyplot as plt
import os
from hpbandster.core.result import Run
import numpy as np

with open('hn_allruns_results_run_9999.json', 'r') as fh:
    all_runs_serialized = json.load(fh)



all_runs = []

for i, v in all_runs_serialized.items():
    # print(v)
    r = Run(config_id=v['config_id'],
            budget=v['budget'],
            loss=v['loss'],
            info=v['info'],
            time_stamps=v['time_stamps'],
            error_logs=None
            )
    all_runs.append(r)
    

# Compare source of:
#     import hpbandster.visualization as hpvis
#     hpvis.losses_over_time(all_runs)
import matplotlib.pyplot as plt
get_loss_from_run_fn = lambda r: r.loss
budgets = set([r.budget for r in all_runs])
data = {}
for b in budgets:
    data[b] = []
for i, r in enumerate(all_runs):
    if r.loss is None:
        continue
    b = r.budget
    # t = r.time_stamps['finished']
    l = get_loss_from_run_fn(r)
    t = i
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
