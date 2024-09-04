import sys
import os
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

Kxs = [[15], [30], [40], [50]]
Runs = [60, 70, 80, 100]

Epochs = 100
num_designs = 100
latent_dim = 4
patience = 50

count = 0
for Kx in Kxs:
    for runs in Runs:
        if runs + 1 < Kx[0]:
            print(f'Run {count} Completed. Execution Time: 0')
            count += 1
            continue
        print(f'Run {count} Started with Kx-{Kx[0]} and Runs-{runs}')
        model = ScalarOnScalarModel(Kx=Kx)
        optimizer = NBDO(model=model, latent_dim=latent_dim)
        start_time = time()
        optimizer.compute_train_set(num_designs=num_designs, runs=runs)
        history = optimizer.fit(epochs=Epochs, patience=patience)
        opt_cr, opt_des = optimizer.optimize()
        end_time = time()
        optimizer.clear_memory()

        run_time = end_time - start_time

        data = {
            "runID": count,
            "model": "ScalarOnScalarModel",
            "runs": runs,
            "Kx": Kx[0],
            "algo": "NBDO",
            "epochs": Epochs,
            "refinement_epochs": None,
            "levels": None,
            "latent_dim": latent_dim,
            "num_layers": optimizer.num_layers,
            "num_designs": num_designs,
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

