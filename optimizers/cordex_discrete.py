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

    def optimize(self, runs, nx, levels, scalars=0, epochs=1000, final_pass_iter=100):
        best_design = None
        best_optimality_value = np.inf

        for _ in tqdm(range(epochs)):
            Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=nx, scalars=scalars)
            model_matrix = Gamma_
            best_design, best_optimality_value = self._evaluate_levels(model_matrix, levels, runs, nx, scalars,
                                                                       best_optimality_value)

        if final_pass_iter > 0:
            for _ in tqdm(range(final_pass_iter)):
                best_design, best_optimality_value = self._evaluate_levels(best_design, levels, runs, nx, scalars,
                                                                           best_optimality_value)

        return best_design, np.abs(best_optimality_value)

    def _evaluate_levels(self, model_matrix, levels, runs, nx, scalars, best_optimality_value):
        current_optimality_value = best_optimality_value

        for i in range(model_matrix.shape[0]):
            for j in range(model_matrix.shape[1]):
                if isinstance(self.model, FunctionOnFunctionModel):
                    objective = lambda x: self.model.compute_objective_input(x, i, j, model_matrix, runs, nx)
                elif isinstance(self.model, ScalarOnFunctionModel):
                    objective = lambda x: self.model.compute_objective_input(x, i, j, model_matrix, sum(nx) + scalars)

                best_lvl_list = [(objective(lvl), lvl) for lvl in levels]
                best_lvl = min(best_lvl_list, key=lambda x: x[0])[1]
                model_matrix[i, j] = best_lvl
                current_optimality_value = objective(model_matrix)

        if 0 <= current_optimality_value < best_optimality_value:
            best_optimality_value = current_optimality_value
            best_design = model_matrix.copy()

        return best_design, best_optimality_value
