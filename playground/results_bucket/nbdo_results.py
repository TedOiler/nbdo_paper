import sys
import os
import numpy as np
import pandas as pd
from time import time
from datetime import timedelta

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/mathematical_models')
from mathematical_models.s_on_s import ScalarOnScalarModel

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/optimizers')
from optimizers.nbdo import NBDO

csv_file_path = "nbdo_results.csv"
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

count = 0
for Kx in Kxs:
    for runs in Runs:
        if runs + 1 < Kx[0]:
            print(f'Run {count} Completed. Execution Time: 0')
            count += 1
            continue
        print(f'Run {count} Started with Kx-{Kx[0]} and Runs-{runs}')
        model = ScalarOnScalarModel(Kx=Kx)
        optimizer = NBDO(model=model, latent_dim=4)
        start_time = time()
        optimizer.compute_train_set(num_designs=1000, runs=runs)
        history = optimizer.fit(epochs=Epochs)
        opt_cr, opt_des = optimizer.optimize()
        end_time = time()
        optimizer.clear_memory()

        run_time = end_time - start_time

        data = {
            "runID": count,
            "model": "ScalarOnScalarModel",
            "runs": runs,
            "Kx": Kx,
            "algo": "NBDO",
            "epochs": Epochs,
            "refinement_epochs": None,
            "levels": None,
            "latent_dim": 4,
            "num_layers": optimizer.num_layers,
            "num_designs": 1000,
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

