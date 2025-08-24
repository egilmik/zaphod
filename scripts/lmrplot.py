import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return 1 + np.log(x) * np.log(y) / 2.5

def main():
    # Create grid
    x = np.linspace(0.1, 20, 400)  # avoid log(0)
    y = np.linspace(0.1, 20, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Define the contour levels we want
    levels = [1, 2, 3, 4, 5, 6]

    # Filled contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=levels, cmap="viridis", extend="both")
    cbar = plt.colorbar(contour, ticks=levels)
    cbar.set_label("f(x, y)")
    
    plt.title("Contour plot of 1 + ln(x)*ln(y)/3")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    main()
