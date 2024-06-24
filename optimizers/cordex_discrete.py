from .base_optimizer import BaseOptimizer
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


class CordexDiscrete(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def optimize(self, runs, nx, epochs=1000, final_pass_iter=100, levels=None):
        best_design, best_optimality_value = None, np.inf

        for _ in tqdm(range(epochs)):
            Gamma = gen_rand_design_m(runs=runs, f_list=nx)
            best_design, best_optimality_value = self._cordex_loop(
                Gamma, runs, nx, best_optimality_value, best_design, levels)

        if final_pass_iter > 0:
            for _ in tqdm(range(final_pass_iter)):
                best_design, best_optimality_value = self._cordex_loop(
                    best_design, runs, nx, best_optimality_value, best_design, levels
                )

        return best_design, np.abs(best_optimality_value)

    def _cordex_loop(self, model_matrix, runs, nx, best_optimality_value, best_design, levels):
        current_optimality_value = best_optimality_value

        for i in range(model_matrix.shape[0]):
            for j in range(model_matrix.shape[1]):
                objective = self._get_objective_function(i, j, model_matrix, runs, nx)
                best_level, best_obj_value = self._evaluate_objective_levels(
                    objective, model_matrix[i, j], levels)

                model_matrix[i, j] = best_level
                current_optimality_value = best_obj_value

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

    def _evaluate_objective_levels(self, objective, current_level, levels):
        best_level = current_level
        best_obj_value = objective(current_level)

        for level in levels:
            obj_value = objective(level)
            if obj_value < best_obj_value:
                best_level, best_obj_value = level, obj_value

        return best_level, best_obj_value
