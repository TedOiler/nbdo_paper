import sys
import os
import numpy as np

sys.path.append(os.path.abspath("../../mathematical_models"))
from mathematical_models.s_on_f import ScalarOnFunctionModel
from mathematical_models.s_on_s import ScalarOnScalarModel

sys.path.append(os.path.abspath("../../optimizers"))
from optimizers.nbdo import NBDO
from optimizers.cordex_continuous import CordexContinuous
from optimizers.cordex_discrete import CordexDiscrete

sys.path.append(os.path.abspath("../../utilities"))
from utilities.plotting.plot_fun import subplot_results


