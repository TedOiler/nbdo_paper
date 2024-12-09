import sys
import os
import numpy as np
sys.path.append(os.path.abspath("../../../mathematical_models"))
from mathematical_models.f_on_f import FunctionOnFunctionModel

sys.path.append(os.path.abspath("../../../optimizers"))
from optimizers.cordex_continuous import CordexContinuous

sys.path.append(os.path.abspath("../../basis"))
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
import csv

parameters_file = 'parameters.csv'
results_file = 'results.csv'
count = 0
hyperparameters_list = []

with open(parameters_file, 'r', newline='') as csvfile:
    # Assuming the parameters.csv is tab-delimited as per your table
    reader = csv.DictReader(csvfile, delimiter='\t')
    for row in reader:
        params = {
            'epochs': int(row['epochs']),
            'refinement': int(row['refinement']),
            'Runs': int(row['Runs']),
            'X_type': row['X type'],
            'X_degree': int(row['X degree']) if row['X degree'] != '-' else None,
            'X_breaks': int(row['X breaks']) if row['X breaks'] != '-' else None,
            'B_type': row['B type'],
            'B_degree': int(row['B degree']) if row['B degree'] != '-' else None,
            'B_breaks': int(row['B breaks']) if row['B breaks'] != '-' else None,
        }
        hyperparameters_list.append(params)

# Open the results.csv file for writing and write the header
with open(results_file, 'w', newline='') as csvfile:
    fieldnames = ['epochs', 'refinement', 'Runs', 'X_type', 'X_degree', 'X_breaks',
                  'B_type', 'B_degree', 'B_breaks', 'design', 'criterion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

total_runs = len(hyperparameters_list)
# Loop over each set of hyperparameters
for params in hyperparameters_list:
    # Extract hyperparameters
    epochs = params['epochs']
    refinement_epochs = params['refinement']
    N = params['Runs']
    X_type = params['X_type']  # Will be 'bsplines'
    X_degree = params['X_degree']
    X_breaks = params['X_breaks']
    B_type = params['B_type']
    B_degree = params['B_degree']
    B_breaks = params['B_breaks']

    try:
        # Define x_base_1 (X_type is always 'bsplines')
        x_base_1 = BSplineBasis(degree=X_degree, total_knots_num=X_breaks)

        # Define b_base_1 based on B_type
        if B_type == 'bsplines':
            b_base_1 = BSplineBasis(degree=B_degree, total_knots_num=B_breaks)
        elif B_type == 'polynomial':
            b_base_1 = PolynomialBasis(degree=B_degree)
        else:
            b_base_1 = None

        # Set up the model and optimizer
        bases_pairs = [(x_base_1, b_base_1)]

        model = FunctionOnFunctionModel(bases_pairs=bases_pairs, Sigma_decay=np.inf)
        optimizer = CordexContinuous(model=model, runs=N)
        best_design, best_objective_value = optimizer.optimize(epochs=epochs, refinement_epochs=refinement_epochs)

        design_str = str(best_design)
        criterion_str = str(best_objective_value)

    except Exception as e:
        design_str = 'NONE'
        criterion_str = 'NONE'
        print(f"Error: on run {count}/{total_runs}")

    with open(results_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        row = {
            'epochs': epochs,
            'refinement': refinement_epochs,
            'Runs': N,
            'X_type': X_type,
            'X_degree': X_degree if X_degree is not None else '-',
            'X_breaks': X_breaks if X_breaks is not None else '-',
            'B_type': B_type,
            'B_degree': B_degree if B_degree is not None else '-',
            'B_breaks': B_breaks if B_breaks is not None else '-',
            'design': design_str,
            'criterion': criterion_str
        }
        writer.writerow(row)
    count += 1
    print(f"Finished run {count}/{total_runs}")

print("All runs completed!")
