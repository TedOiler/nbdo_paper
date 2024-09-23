import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skopt import gp_minimize
from skopt.plots import plot_convergence

# Define a smoother, non-symmetric objective function (noise-free)
def objective_function(x):
    return (x[0] ** 2 + x[1] ** 2) + 0.5 * np.sin(2 * np.pi * x[0]) + 0.25 * (x[1] + 0.5) ** 2

# Define the bounds for L1 and L2
bounds = [(-1.0, 1.0), (-1.0, 1.0)]

# Run Bayesian Optimization for 10 steps
res = gp_minimize(objective_function, dimensions=bounds, n_calls=10, random_state=42)

# Prepare the grid for the 3D surface plot
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = objective_function([X, Y])

# Create the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

# Overlay the Bayesian optimization points
x_coords = [res.x_iters[i][0] for i in range(len(res.x_iters))]
y_coords = [res.x_iters[i][1] for i in range(len(res.x_iters))]
z_coords = [objective_function([x_coords[i], y_coords[i]]) for i in range(len(res.x_iters))]

# Add a small offset to z_coords to ensure points and lines stay above the surface
z_offset = 2  # You can adjust this value to raise the points higher if needed
z_coords_above = [z + z_offset for z in z_coords]

# Plot the red points with larger size, ensuring they are above the surface
ax.scatter(x_coords, y_coords, z_coords_above, color='red', s=100, alpha=1.0)

# Plot the connecting line between the points, ensuring it is above the surface
ax.plot(x_coords, y_coords, z_coords_above, color='black', linestyle='-', linewidth=2)

# Highlight the minimum point found by BO
x_min = res.x[0]  # x-coordinate of the minimum
y_min = res.x[1]  # y-coordinate of the minimum
z_min = objective_function([x_min, y_min]) + z_offset  # z-coordinate with the offset

# Plot the minimum point as a larger yellow point on top of the red point
ax.scatter(x_min, y_min, z_min, color='black', s=200, marker='o', edgecolors='none', alpha=1.0)

# Set the viewing angle (elevation and azimuth)
ax.view_init(elev=70, azim=-90)  # Adjust the viewing angle as needed

ax.legend()

# Save the figure
plt.savefig("latent_space_with_minimum.png")

# Display the plot
plt.show()

# Output the coordinates of the minimum
print(np.round(x_min, 2), np.round(y_min, 2))
#%%
