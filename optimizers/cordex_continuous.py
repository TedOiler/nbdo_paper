from .base_optimizer import BaseOptimizer
from scipy.optimize import minimize
import numpy as np
import sys
from pathlib import Path
from utilities.help.gen_rand_design import gen_rand_design
from mathematical_models.f_on_f import FunctionOnFunctionModel
from mathematical_models.s_on_f import ScalarOnFunctionModel
from mathematical_models.s_on_s import ScalarOnScalarModel
from tqdm import tqdm

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))


class CordexContinuous(BaseOptimizer):
    def __init__(self, model, runs):
        super().__init__(model)
        self.model = model
        self.runs = runs

    def optimize(self, epochs: int = 1000, refinement_epochs: int = 100) -> tuple:

        if self.runs < self.model.Kb + 1:
            raise ValueError(f"Model not estimable with {self.runs} runs and {self.model.Kb} parameters."
                             f" Need at least {self.model.Kb + 1} runs.")

        best_Gamma, best_optimality_value = None, np.inf

        for _ in tqdm(range(epochs)):
            Gamma = gen_rand_design(N=self.runs, P=self.model.Kx)
            best_Gamma, best_optimality_value = self._cordex_loop(Gamma, self.runs, self.model.Kx,
                                                                  best_optimality_value, best_Gamma)

        if refinement_epochs > 0:
            for _ in tqdm(range(refinement_epochs)):
                best_Gamma, best_optimality_value = self._cordex_loop(best_Gamma, self.runs, self.model.Kx,
                                                                      best_optimality_value, best_Gamma)

        return best_Gamma, np.abs(best_optimality_value)

    def _cordex_loop(self, Gamma, runs, Kx, best_optimality_value, best_design):
        current_optimality_value = best_optimality_value
        for i in range(Gamma.shape[0]):
            for j in range(Gamma.shape[1]):
                objective = self._get_objective_function(i, j, Gamma, runs, Kx)
                result = minimize(objective, Gamma[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                if result.x is not None:
                    Gamma[i, j] = result.x
                current_optimality_value = objective(result.x)

        if 0 <= current_optimality_value < best_optimality_value:
            best_optimality_value = current_optimality_value
            best_design = Gamma.copy()

        return best_design, best_optimality_value

    def _get_objective_function(self, i, j, Gamma, runs, Kx):
        if isinstance(self.model, FunctionOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, Gamma, runs, Kx)
        elif isinstance(self.model, ScalarOnFunctionModel):
            return lambda x: self.model.compute_objective_input(x, i, j, Gamma)
        elif isinstance(self.model, ScalarOnScalarModel):
            return lambda x: self.model.compute_objective_input(x, i, j, Gamma)
        else:
            raise TypeError("Unsupported model type")
