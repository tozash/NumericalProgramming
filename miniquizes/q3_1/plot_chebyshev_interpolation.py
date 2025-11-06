"""
Chebyshev Interpolation on [-1, 1] with n=3 nodes
Plots the original function, Chebyshev nodes, and Newton interpolating polynomial
"""

import numpy as np
import matplotlib.pyplot as plt


def divided_differences(x, y):
    """
    Compute Newton's divided difference coefficients.
    
    Parameters:
    -----------
    x : array-like
        The nodes (sorted ascending)
    y : array-like
        The function values at nodes
    
    Returns:
    --------
    coeffs : ndarray
        Array of Newton coefficients [f[x_0], f[x_0,x_1], f[x_0,x_1,x_2]]
    """
    n = len(x)
    # Initialize divided difference table
    table = np.zeros((n, n))
    table[:, 0] = y  # First column is f[x_i] = y_i
    
    # Fill the divided difference table
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = (table[i + 1, j - 1] - table[i, j - 1]) / (x[i + j] - x[i])
    
    # Return the top row (coefficients for Newton form)
    return table[0, :]


def newton_eval(x_nodes, coeffs, x_eval):
    """
    Evaluate Newton polynomial at given points.
    
    Parameters:
    -----------
    x_nodes : array-like
        The nodes used for interpolation
    coeffs : array-like
        Newton coefficients from divided_differences
    x_eval : array-like or scalar
        Points at which to evaluate the polynomial
    
    Returns:
    --------
    y_eval : ndarray or scalar
        Evaluated polynomial values
    """
    x_eval = np.asarray(x_eval)
    n = len(coeffs)
    result = np.zeros_like(x_eval, dtype=float)
    
    # Horner's method for Newton form
    for i in range(n - 1, -1, -1):
        result = result * (x_eval - x_nodes[i]) + coeffs[i]
    
    return result


def main():
    """Main function to compute and plot Chebyshev interpolation."""
    
    # Number of nodes
    n = 3
    
    # Compute Chebyshev nodes on [-1, 1]
    # t_j = cos((2j-1)π/2n) for j = 1, 2, ..., n
    j_indices = np.arange(1, n + 1)
    chebyshev_unsorted = np.cos((2 * j_indices - 1) * np.pi / (2 * n))
    
    # Sort ascending: x_0 < x_1 < x_2
    x_nodes = np.sort(chebyshev_unsorted)
    
    # Define function f(x) = sin²(x) * cos(x)
    def f(x):
        return (np.sin(x)**2) * np.cos(x)
    
    # Evaluate function at nodes
    y_nodes = f(x_nodes)
    
    # Compute divided differences
    coeffs = divided_differences(x_nodes, y_nodes)
    
    # Create dense grid for plotting
    xx = np.linspace(-1, 1, 2000)
    
    # Evaluate original function and polynomial on grid
    f_xx = f(xx)
    P2_xx = newton_eval(x_nodes, coeffs, xx)
    
    # Evaluate P2 at x = 0.5
    P2_at_0_5 = newton_eval(x_nodes, coeffs, 0.5)
    
    # Print results to console
    print("Sorted Chebyshev nodes:")
    print(x_nodes)
    print("\nFunction values at nodes:")
    print(y_nodes)
    print("\nDivided-difference coefficients:")
    print(coeffs)
    print(f"\nP₂(0.5) = {P2_at_0_5:.10f}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot original function
    plt.plot(xx, f_xx, linewidth=2.5, label=r'$f(x) = \sin^2(x) \cos(x)$')
    
    # Plot nodes
    plt.scatter(x_nodes, y_nodes, s=50, marker='o', color='red', 
                zorder=5, label='Chebyshev nodes (n=3)')
    
    # Plot Newton polynomial
    plt.plot(xx, P2_xx, '--', linewidth=2, label=r'$P_2(x)$ (Newton, n=3)')
    
    # Formatting
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Chebyshev Interpolation on [−1,1] (n=3)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figures
    plt.savefig('chebyshev_n3_plot.png', dpi=200, bbox_inches='tight')
    plt.savefig('chebyshev_n3_plot.pdf', bbox_inches='tight')
    print("\nFigures saved as chebyshev_n3_plot.png and chebyshev_n3_plot.pdf")
    
    # Display plot
    plt.show()


if __name__ == '__main__':
    main()

