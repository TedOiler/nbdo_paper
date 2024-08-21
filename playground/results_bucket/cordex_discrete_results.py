import sys
import os
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/mathematical_models')
from mathematical_models.s_on_s import ScalarOnScalarModel

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/optimizers')
from optimizers.cordex_discrete import CordexDiscrete

csv_file_path = "cordex_discrete_results.csv"
columns = ["runID",
           "model",
           "runs",
           "Kx",
           "algo",
           "epochs",
           "refinement_epochs",
           "levels",
           "latent_dim",
           "num_layers",
           "num_designs",
           "opt_cr",
           "opt_des",
           "run_time_s",
           "run_time_f", ]
file_exists = os.path.isfile(csv_file_path)
df = pd.DataFrame(columns=columns)

Kxs = [[4], [8], [12], [20], [40]]
Runs = [5, 15, 30, 50, 60]
Epochs = 1000
Refinement_epochs = 100
Levels = [np.linspace(-1., 1., 2), np.linspace(-1., 1., 3), np.linspace(-1., 1., 5)]

count = 0
for Kx in Kxs:
    for runs in Runs:
        if runs + 1 < Kx[0]:
            print(f'Run {count} Completed. Execution Time: 0')
            count += 1
            continue
        for lvls in Levels:
            print(f'Run {count} Started with Kx-{Kx[0]} and Runs-{runs}')
            model = ScalarOnScalarModel(Kx=Kx)
            optimizer = CordexDiscrete(model=model, levels=lvls, runs=runs)
            start_time = time()
            opt_des, opt_cr = optimizer.optimize(epochs=Epochs, refinement_epochs=Refinement_epochs)
            end_time = time()
            run_time = end_time - start_time

            data = {
                "runID": count,
                "model": "ScalarOnScalarModel",
                "runs": runs,
                "Kx": Kx,
                "algo": "CordexDiscrete",
                "epochs": Epochs,
                "refinement_epochs": Refinement_epochs,
                "levels": lvls,
                "latent_dim": None,
                "num_layers": None,
                "num_designs": None,
                "opt_cr": opt_cr,
                "opt_des": opt_des,
                "run_time_s": run_time,
                "run_time_f": timedelta(seconds=run_time)
            }
            new_row = pd.DataFrame([data])
            new_row.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)
            file_exists = True  # Ensure header is not written again
            print(f'Run {count} Completed. Execution Time: {timedelta(seconds=run_time)}')
            count += 1


print(f"Results saved to {csv_file_path}")
