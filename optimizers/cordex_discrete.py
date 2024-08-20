from .base_optimizer import BaseOptimizer
import numpy as np
import sys
from pathlib import Path
from utilities.help.gen_rand_design import gen_rand_design_m
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel
from mathematical_models.s_on_s import ScalarOnScalarModel
from tqdm import tqdm

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))


class CordexDiscrete(BaseOptimizer):
    def __init__(self, model, runs, levels):
        super().__init__(model)
        self.model = model
        self.runs = runs
        self.levels = levels

    def optimize(self, epochs=1000, refinement_iterations=100):
        best_design, best_optimality_value = None, np.inf

        for _ in tqdm(range(epochs)):
            Gamma = gen_rand_design_m(runs=self.runs, f_list=self.model.Kx)
            best_design, best_optimality_value = self._cordex_loop(
                Gamma, best_optimality_value, best_design)

        if refinement_iterations > 0:
            for _ in tqdm(range(refinement_iterations)):
                best_design, best_optimality_value = self._cordex_loop(
                    best_design, best_optimality_value, best_design)

        return best_design, np.abs(best_optimality_value)

    def _cordex_loop(self, model_matrix, best_optimality_value, best_design):
        current_optimality_value = best_optimality_value

        for i in range(model_matrix.shape[0]):
            for j in range(model_matrix.shape[1]):
                objective = self._get_objective_function(i, j, model_matrix)
                best_level, best_obj_value = self._evaluate_objective_levels(
                    objective, model_matrix[i, j])

                model_matrix[i, j] = best_level
                current_optimality_value = best_obj_value

        if 0 <= current_optimality_value < best_optimality_value:
            best_optimality_value = current_optimality_value
            best_design = model_matrix.copy()

        return best_design, best_optimality_value

    def _get_objective_function(self, i, j, model_matrix):
        if isinstance(self.model, FunctionOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, model_matrix, self.runs, self.model.Kx)
        elif isinstance(self.model, ScalarOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, model_matrix, sum(self.model.Kx))
        elif isinstance(self.model, ScalarOnScalarModel):
            return lambda x: self.model.compute_objective_input(x, i, j, model_matrix)
        else:
            raise TypeError("Unsupported model type")

    def _evaluate_objective_levels(self, objective, current_level):
        best_level = current_level
        best_obj_value = objective(current_level)

        for level in self.levels:
            obj_value = objective(level)
            if obj_value < best_obj_value:
                best_level, best_obj_value = level, obj_value

        return best_level, best_obj_value
