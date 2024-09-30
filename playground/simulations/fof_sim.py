import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.stats import multivariate_normal
from skfda.representation.basis import BSplineBasis
from skfda.representation.grid import FDataGrid
from typing import Tuple, Optional


### Helper Functions ###

def evaluate_bifd(eval_x: np.ndarray, eval_y: np.ndarray, bifdobj: FDataGrid) -> np.ndarray:
    """
    Function to evaluate a bifunctional object at given points.
    """
    return bifdobj(eval_x, eval_y)


def evaluate_fd(eval_x: np.ndarray, fdobj: FDataGrid) -> np.ndarray:
    """
    Function to evaluate a functional object at given points.
    """
    return fdobj(eval_x)


def compute_trapz_integration(B_column: np.ndarray, x_values: np.ndarray, eval_x: np.ndarray) -> float:
    """
    Compute the trapezoidal integration for a given column of B and functional values x.
    """
    return np.trapz(B_column * x_values, eval_x)


### Convolution Function ###

def compute_convolution(fdobj: FDataGrid, bifdobj: FDataGrid,
                        eval_x: Optional[np.ndarray] = None,
                        eval_y: Optional[np.ndarray] = None) -> FDataGrid:
    """
    Compute the convolution between an fd object and a bifd object.
    """
    if eval_x is None:
        eval_x = np.linspace(fdobj.basis.domain_range[0][0], fdobj.basis.domain_range[0][1], 100)
    if eval_y is None:
        eval_y = np.linspace(bifdobj.basis.domain_range[0][0], bifdobj.basis.domain_range[0][1], 100)

    # Set quadrature points and weights
    xmin: float = fdobj.basis.domain_range[0][0]
    xmax: float = fdobj.basis.domain_range[0][1]
    n_quad_points: int = len(eval_x)

    # Initialize array to store convolution results
    convolution_y: np.ndarray = np.empty(len(eval_y))

    # Evaluate the bifd and fd objects at the quadrature points
    B: np.ndarray = evaluate_bifd(eval_x, eval_y, bifdobj)
    x: np.ndarray = evaluate_fd(eval_x, fdobj)

    # Compute the convolution numerically
    for k in range(len(convolution_y)):
        convolution_y[k] = compute_trapz_integration(B[:, k], x, eval_x)

    return FDataGrid(data_matrix=convolution_y, grid_points=eval_y)


### Radial basis Function Kernel (RBF) ###

def RBF(x1: np.ndarray, x2: np.ndarray, par: Tuple[float, float]) -> float:
    """
    Radial basis Function (RBF) kernel for Gaussian Process.
    """
    h: float = np.dot((x1 - x2).T, (x1 - x2)) / par[1]
    return par[0] * np.exp(-h / 2)


### Gaussian Process Simulation ###

def simulate_gaussian_process(gamma: float = 1, sigma: float = 1,
                              range_x: Tuple[float, float] = (0, 1),
                              nbasis: int = 100) -> FDataGrid:
    """
    Simulate Gaussian Process error using RBF kernel.
    """
    n: int = 100
    x_grid: np.ndarray = np.linspace(range_x[0], range_x[1], n)
    Xgrid: np.ndarray = x_grid[:, np.newaxis]

    # Compute covariance matrix using RBF kernel
    Kmat: np.ndarray = np.array([[RBF(Xgrid[i], Xgrid[j], [gamma, sigma]) for j in range(n)] for i in range(n)])

    # Simulate multivariate normal with the given covariance matrix
    y: np.ndarray = multivariate_normal.rvs(mean=np.zeros(n), cov=Kmat)

    return FDataGrid(data_matrix=y - np.mean(y), grid_points=x_grid)


### Plotting Functions ###

def plot_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, title: str = '3D Surface Plot') -> None:
    """
    Plot a 3D surface for given X, Y, and Z values.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    plt.title(title)
    plt.show()


def plot_fd_object(fd_obj: FDataGrid) -> None:
    """
    Plot a functional data object.
    """
    fd_obj.plot()


### Coefficient and basis Functions Setup ###

def setup_true_coefficient_beta(n_s: int, n_t: int) -> FDataGrid:
    """
    Setup the true coefficient beta as a bifunctional object.
    """
    spline_basis_s: BSplineBasis = BSplineBasis(domain_range=(0, 1), n_basis=n_s, order=4)
    spline_basis_t: BSplineBasis = BSplineBasis(domain_range=(0, 1), n_basis=n_t, order=4)
    beta_coef: np.ndarray = np.random.normal(size=(n_s, n_t))

    return FDataGrid(beta_coef, grid_points=[spline_basis_s.linspace(n_s), spline_basis_t.linspace(n_t)])


def setup_basis_for_factor(nxbasis: int) -> BSplineBasis:
    """
    Setup basis for the factor using B-Spline basis.
    """
    return BSplineBasis(domain_range=(0, 1), n_basis=nxbasis, order=1)


def generate_factor_coefficients(run: int, nxbasis: int) -> np.ndarray:
    """
    Generate coefficients for the factor (random values).
    """
    return np.random.uniform(0, 1, size=(run, nxbasis))


def setup_factor_function(Xcoeff: np.ndarray, xbasis: BSplineBasis) -> FDataGrid:
    """
    Setup the factor as a functional data object.
    """
    return FDataGrid(data_matrix=Xcoeff.T, grid_points=xbasis.linspace(len(Xcoeff[0])))


### Simulation of Responses ###

def simulate_responses(run: int, x1: FDataGrid, beta: FDataGrid,
                       noise_level: float, noise_basis: int, n_t: int) -> Tuple[FDataGrid, FDataGrid]:
    """
    Simulate responses by adding Gaussian Process noise to the true responses.
    """
    Ycoeff: np.ndarray = np.empty((run, 83))
    trueYcoeff: np.ndarray = np.empty((run, n_t))

    for j in range(run):
        xj: FDataGrid = FDataGrid(data_matrix=x1.data_matrix[:, j], grid_points=x1.grid_points[0])
        z: FDataGrid = compute_convolution(xj, beta)
        eps: FDataGrid = simulate_gaussian_process(gamma=noise_level, sigma=0.01, range_x=z.grid_points[0],
                                                   nbasis=noise_basis)
        y: FDataGrid = z + eps
        Ycoeff[j, :] = y.data_matrix
        trueYcoeff[j, :] = z.data_matrix

    Y: FDataGrid = FDataGrid(data_matrix=Ycoeff.T, grid_points=z.grid_points[0])
    trueY: FDataGrid = FDataGrid(data_matrix=trueYcoeff.T, grid_points=beta.grid_points[1])

    return Y, trueY


### Main Execution ###

if __name__ == "__main__":
    # Number of basis functions
    n_s: int = 4
    n_t: int = 4

    # Setup true coefficient beta
    beta: FDataGrid = setup_true_coefficient_beta(n_s, n_t)

    # Plot 3D surface of the true coefficient beta
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    X, Y = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
    Z = beta(X, Y)
    plot_3d_surface(X, Y, Z, title='True Coefficient Beta')

    # Setup basis and coefficients for the factor
    nxbasis: int = 10
    xbasis: BSplineBasis = setup_basis_for_factor(nxbasis)
    run: int = 200
    Xcoeff: np.ndarray = generate_factor_coefficients(run, nxbasis)

    # Setup factor function
    x1: FDataGrid = setup_factor_function(Xcoeff, xbasis)

    # Plot the factor function
    plot_fd_object(x1)

    # Simulate responses
    noise_basis: int = 80
    noise_level: float = 0.005
    Y: FDataGrid
    trueY: FDataGrid
    Y, trueY = simulate_responses(run, x1, beta, noise_level, noise_basis, n_t)

    # Plot the simulated responses
    plot_fd_object(Y)
    plot_fd_object(trueY)
