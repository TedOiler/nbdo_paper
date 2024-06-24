from .base_optimizer import BaseOptimizer
from scipy.optimize import minimize
import numpy as np
import sys
from pathlib import Path
from utilities.gen_rand_design import gen_rand_design_m
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel
from tqdm import tqdm

# Setting up the directory paths
current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))


class CordexContinuous(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def optimize(self, runs, nx, epochs=1000, final_pass_iter=100):
        best_design, best_optimality_value = None, np.inf

        for _ in tqdm(range(epochs)):
            Gamma = gen_rand_design_m(runs=runs, f_list=nx)
            best_design, best_optimality_value = self._cordex_loop(Gamma, runs, nx,
                                                                   best_optimality_value, best_design)

        if final_pass_iter > 0:
            for _ in tqdm(range(final_pass_iter)):
                best_design, best_optimality_value = self._cordex_loop(best_design, runs, nx,
                                                                       best_optimality_value, best_design)

        return best_design, np.abs(best_optimality_value)

    def _cordex_loop(self, model_matrix, runs, nx, best_optimality_value, best_design):
        current_optimality_value = best_optimality_value

        for i in range(model_matrix.shape[0]):
            for j in range(model_matrix.shape[1]):
                objective = self._get_objective_function(i, j, model_matrix, runs, nx)
                result = minimize(objective, model_matrix[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                if result.x is not None:
                    model_matrix[i, j] = result.x
                current_optimality_value = objective(result.x)

        if 0 <= current_optimality_value < best_optimality_value:
            best_optimality_value = current_optimality_value
            best_design = model_matrix.copy()

        return best_design, best_optimality_value

    def _get_objective_function(self, i, j, model_matrix, runs, nx):
        if isinstance(self.model, FunctionOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, model_matrix, runs, nx)
        elif isinstance(self.model, ScalarOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, model_matrix, sum(nx))
        else:
            raise TypeError("Unsupported model type")
