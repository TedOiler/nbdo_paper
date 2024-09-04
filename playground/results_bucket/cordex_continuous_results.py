import sys
import os
import pandas as pd
from time import time
from datetime import timedelta

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/mathematical_models')
from mathematical_models.s_on_s import ScalarOnScalarModel

sys.path.append('/Users/thodorisladas/Documents/code/system_optidex/optimizers')
from optimizers.cordex_continuous import CordexContinuous

csv_file_path = "cordex_continuous_results.csv"
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

Kxs = [[15], [30], [40], [50]]
Runs = [60, 70, 80, 100]
Epochs = 10
Refinement_epochs = 1

count = 0
for Kx_index, Kx in enumerate(Kxs):
    for run_index, runs in enumerate(Runs):
        if runs + 1 < Kx[0]:
            print(f'Run {count} Completed. Execution Time: 0')
            count += 1
            continue
        print(f'Run {count} Started with Kx-{Kx[0]} and Runs-{runs}')
        model = ScalarOnScalarModel(Kx=Kx)
        optimizer = CordexContinuous(model=model, runs=runs)

        start_time = time()
        opt_des, opt_cr = optimizer.optimize(epochs=Epochs, refinement_epochs=Refinement_epochs)
        end_time = time()
        run_time = end_time - start_time

        data = {
            "runID": count,
            "model": "ScalarOnScalarModel",
            "runs": runs,
            "Kx": Kx[0],
            "algo": "CordexContinuous",
            "epochs": Epochs,
            "refinement_epochs": Refinement_epochs,
            "levels": None,
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
