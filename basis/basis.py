import numpy as np
import matplotlib.pyplot as plt


class Basis:
    def evaluate_basis_function(self, i, t):
        raise NotImplementedError("The method evaluate_basis_function must be implemented by the subclass.")

    def get_basis_support(self, i):
        raise NotImplementedError("The method get_basis_support must be implemented by the subclass.")

    def num_basis(self):
        raise NotImplementedError("The method num_basis must be implemented by the subclass.")

    def evaluate_combination(self, coefficients, t_values):
        num_basis = self.num_basis()
        if len(coefficients) != num_basis:
            raise ValueError("Length of coefficients must match number of basis functions.")
        result = np.zeros_like(t_values, dtype=float)
        for i in range(num_basis):
            basis_values = np.array([self.evaluate_basis_function(i, t) for t in t_values])
            result += coefficients[i] * basis_values
        return result

    def plot_basis_functions(self, t_values):
        num_basis = self.num_basis()
        plt.style.use('fivethirtyeight')
        for i in range(num_basis):
            basis_values = np.array([self.evaluate_basis_function(i, t) for t in t_values])
            plt.plot(t_values, basis_values)
        plt.title(f'{self.__class__.__name__} Functions')
        plt.xlabel('t')
        plt.ylabel('basis function value')
        plt.grid(False)
        plt.ylim(-1.2, 1.2)
        plt.show()

    def plot_experimental_run(self, t_values, coefficients):
        s_values_fourier = self.evaluate_combination(coefficients, t_values)

        # Plot the combined Fourier function
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8, 6))
        plt.plot(t_values, s_values_fourier)
        plt.title(f'Experimental run of {self.__class__.__name__}')
        plt.xlabel('t')
        plt.ylabel('Combined Function Value')
        plt.grid(False)
        plt.ylim(-1.2, 1.2)
        plt.show()
