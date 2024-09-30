# app/app.py

import sys
import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set the page configuration to 'wide' to utilize the full width of the browser
st.set_page_config(page_title="Optimization Design Explorer", layout="wide")

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the root directory of the project
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the root directory to sys.path
sys.path.insert(0, root_dir)

# Import your modules
from mathematical_models.s_on_f import ScalarOnFunctionModel
from optimizers.cordex_continuous import CordexContinuous
from basis.bspline import BSplineBasis
from basis.polynomial import PolynomialBasis
from basis.fourier import FourierBasis
from basis.basis import plot_design


def main():
    st.title("Optimization Design Explorer")

    # Algorithm Parameters in a Horizontal Container
    st.header("Algorithm Settings")
    with st.container():
        # Create a single row with four equally spaced columns
        col_alg1, col_alg2, col_alg3, col_alg4 = st.columns(4)
        with col_alg1:
            # Select optimizer
            optimizer_option = st.selectbox(
                "Select Optimizer",
                ("Coordinate Exchange Continuous",),
                key="optimizer_option"
            )
        with col_alg2:
            # Number of Runs (N)
            N = st.number_input("Number of Runs (N)", min_value=1, value=6, step=1, key="N_runs")
        with col_alg3:
            # Epochs
            epochs = st.number_input("Epochs", min_value=1, value=100, step=1, key="epochs")
            sub_x = st.number_input("sub_x", min_value=1, value=3, step=1, key="sub_x")

        with col_alg4:
            # Refinement Epochs
            refinement_epochs = st.number_input("Refinement Epochs", min_value=1, value=10, step=1,
                                                key="refinement_epochs")
            sub_y = st.number_input("sub_y", min_value=1, value=3, step=1, key="sub_y")

    st.markdown("---")  # Separator line

    # Two Side-by-Side Containers for Model Settings
    col_model1, col_model2 = st.columns(2)

    with col_model1:
        st.header("Model 1: Scalar on Function")
        model1_option = st.selectbox(
            "Select Mathematical Model for Model 1",
            ("Scalar on Function",),
            key="model1_option"
        )

        # Basis configuration for Model 1
        st.subheader("Basis Configuration for Model 1")
        num_bases_model1 = st.number_input("Number of Basis Pairs", min_value=1, value=1, step=1,
                                           key="num_bases_model1")

        bases_pairs_model1 = []
        x_bases_model1 = []

        for i in range(num_bases_model1):
            st.markdown(f"**Basis Pair {i + 1}**")

            # Create subcolumns for X Basis and B Basis
            subcol_x, subcol_b = st.columns(2)

            with subcol_x:
                st.markdown("**🔍 X Basis Configuration**")
                # X Basis
                x_basis_type = st.selectbox(
                    f"X Basis Type for Pair {i + 1}",
                    ("BSpline", "Fourier", "Polynomial"),
                    key=f"model1_x_basis_type_{i}"
                )

                if x_basis_type == "BSpline":
                    degree = st.number_input(
                        f"X BSpline Degree", min_value=0, value=1, step=1, key=f"model1_x_bspline_degree_{i}"
                    )
                    num_basis_functions = st.number_input(
                        f"X BSpline Basis Functions", min_value=1, value=10, step=1,
                        key=f"model1_x_bspline_num_{i}"
                    )
                    x_basis = BSplineBasis(degree=degree, num_basis_functions=num_basis_functions)
                elif x_basis_type == "Fourier":
                    num_basis_functions = st.number_input(
                        f"X Fourier Basis Functions", min_value=1, value=10, step=1,
                        key=f"model1_x_fourier_num_{i}"
                    )
                    x_basis = FourierBasis(num_basis_functions=num_basis_functions)
                else:  # Polynomial
                    degree = st.number_input(
                        f"X Polynomial Degree", min_value=1, value=1, step=1, key=f"model1_x_poly_degree_{i}"
                    )
                    x_basis = PolynomialBasis(degree=degree)

            with subcol_b:
                st.markdown("**B Basis Configuration**")
                # B Basis
                b_basis_type = st.selectbox(
                    f"B Basis Type for Pair {i + 1}",
                    ("BSpline", "Fourier", "Polynomial"),
                    key=f"model1_b_basis_type_{i}"
                )

                if b_basis_type == "BSpline":
                    degree = st.number_input(
                        f"B BSpline Degree", min_value=0, value=1, step=1, key=f"model1_b_bspline_degree_{i}"
                    )
                    num_basis_functions = st.number_input(
                        f"B BSpline Basis Functions", min_value=1, value=10, step=1,
                        key=f"model1_b_bspline_num_{i}"
                    )
                    b_basis = BSplineBasis(degree=degree, num_basis_functions=num_basis_functions)
                elif b_basis_type == "Fourier":
                    num_basis_functions = st.number_input(
                        f"B Fourier Basis Functions", min_value=1, value=10, step=1,
                        key=f"model1_b_fourier_num_{i}"
                    )
                    b_basis = FourierBasis(num_basis_functions=num_basis_functions)
                else:  # Polynomial
                    degree = st.number_input(
                        f"B Polynomial Degree", min_value=1, value=2, step=1, key=f"model1_b_poly_degree_{i}"
                    )
                    b_basis = PolynomialBasis(degree=degree)

            bases_pairs_model1.append((x_basis, b_basis))
            x_bases_model1.append(x_basis)

    with col_model2:
        st.header("Model 2: Scalar on Function")
        model2_option = st.selectbox(
            "Select Mathematical Model for Model 2",
            ("Scalar on Function",),
            key="model2_option"
        )

        # Basis configuration for Model 2
        st.subheader("Basis Configuration for Model 2")
        num_bases_model2 = st.number_input("Number of Basis Pairs", min_value=1, value=1, step=1,
                                           key="num_bases_model2")

        bases_pairs_model2 = []
        x_bases_model2 = []

        for i in range(num_bases_model2):
            st.markdown(f"**Basis Pair {i + 1}**")

            # Create subcolumns for X Basis and B Basis
            subcol_x, subcol_b = st.columns(2)

            with subcol_x:
                st.markdown("**X Basis Configuration**")
                # X Basis
                x_basis_type = st.selectbox(
                    f"X Basis Type for Pair {i + 1}",
                    ("BSpline", "Fourier", "Polynomial"),
                    key=f"model2_x_basis_type_{i}"
                )

                if x_basis_type == "BSpline":
                    degree = st.number_input(
                        f"X BSpline Degree", min_value=0, value=1, step=1, key=f"model2_x_bspline_degree_{i}"
                    )
                    num_basis_functions = st.number_input(
                        f"X BSpline Basis Functions", min_value=1, value=10, step=1,
                        key=f"model2_x_bspline_num_{i}"
                    )
                    x_basis = BSplineBasis(degree=degree, num_basis_functions=num_basis_functions)
                elif x_basis_type == "Fourier":
                    num_basis_functions = st.number_input(
                        f"X Fourier Basis Functions", min_value=1, value=10, step=1,
                        key=f"model2_x_fourier_num_{i}"
                    )
                    x_basis = FourierBasis(num_basis_functions=num_basis_functions)
                else:  # Polynomial
                    degree = st.number_input(
                        f"X Polynomial Degree", min_value=1, value=1, step=1, key=f"model2_x_poly_degree_{i}"
                    )
                    x_basis = PolynomialBasis(degree=degree)

            with subcol_b:
                st.markdown("**B Basis Configuration**")
                # B Basis
                b_basis_type = st.selectbox(
                    f"B Basis Type for Pair {i + 1}",
                    ("BSpline", "Fourier", "Polynomial"),
                    key=f"model2_b_basis_type_{i}"
                )

                if b_basis_type == "BSpline":
                    degree = st.number_input(
                        f"B BSpline Degree", min_value=0, value=1, step=1, key=f"model2_b_bspline_degree_{i}"
                    )
                    num_basis_functions = st.number_input(
                        f"B BSpline Basis Functions", min_value=1, value=10, step=1,
                        key=f"model2_b_bspline_num_{i}"
                    )
                    b_basis = BSplineBasis(degree=degree, num_basis_functions=num_basis_functions)
                elif b_basis_type == "Fourier":
                    num_basis_functions = st.number_input(
                        f"B Fourier Basis Functions", min_value=1, value=10, step=1,
                        key=f"model2_b_fourier_num_{i}"
                    )
                    b_basis = FourierBasis(num_basis_functions=num_basis_functions)
                else:  # Polynomial
                    degree = st.number_input(
                        f"B Polynomial Degree", min_value=1, value=2, step=1, key=f"model2_b_poly_degree_{i}"
                    )
                    b_basis = PolynomialBasis(degree=degree)

            bases_pairs_model2.append((x_basis, b_basis))
            x_bases_model2.append(x_basis)

    st.markdown("---")  # Separator line

    # Run Optimization Button
    if st.button("🚀 Run Optimization"):
        with st.spinner("🔄 Running optimization..."):
            # Model 1 Optimization
            try:
                if model1_option == "Scalar on Function":
                    model1 = ScalarOnFunctionModel(bases_pairs=bases_pairs_model1)
                else:
                    st.error("❌ Invalid model selection for Model 1.")
                    st.stop()

                if optimizer_option == "Coordinate Exchange Continuous":
                    optimizer1 = CordexContinuous(model=model1, runs=N)
                else:
                    st.error("❌ Invalid optimizer selection.")
                    st.stop()

                best_design1, best_objective_value1 = optimizer1.optimize(
                    epochs=epochs, refinement_epochs=refinement_epochs
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred during optimization of Model 1: {e}")
                st.stop()

            # Model 2 Optimization
            try:
                if model2_option == "Scalar on Function":
                    model2 = ScalarOnFunctionModel(bases_pairs=bases_pairs_model2)
                else:
                    st.error("❌ Invalid model selection for Model 2.")
                    st.stop()

                if optimizer_option == "Coordinate Exchange Continuous":
                    optimizer2 = CordexContinuous(model=model2, runs=N)
                else:
                    st.error("❌ Invalid optimizer selection.")
                    st.stop()

                best_design2, best_objective_value2 = optimizer2.optimize(
                    epochs=epochs, refinement_epochs=refinement_epochs
                )
            except Exception as e:
                st.error(f"⚠️ An error occurred during optimization of Model 2: {e}")
                st.stop()

            st.success("✅ Optimization completed!")

            # Display Plots
            st.header("📈 Optimized Designs")
            col_plot1, col_plot2 = st.columns(2)

            with col_plot1:
                st.subheader("🔍 Model 1")
                fig1 = plot_design(best_design1, x_bases_model1, N, sub_x=sub_x, sub_y=sub_y)
                st.pyplot(fig1)

            with col_plot2:
                st.subheader("🔍 Model 2")
                fig2 = plot_design(best_design2, x_bases_model2, N, sub_x=sub_x, sub_y=sub_y)
                st.pyplot(fig2)


if __name__ == "__main__":
    main()
