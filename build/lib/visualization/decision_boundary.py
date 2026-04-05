import numpy as np
import matplotlib.pyplot as plt

class DecisionBoundaryPlotter:
    def plot(self, model, scaler, X, y):
        original = scaler.inverse_transform(X)

        xmin = original[:,0].min() - 0.1
        xmax = original[:,0].max() + 0.1
        ymin = original[:,1].min() - 0.1
        ymax = original[:,1].max() + 0.1
        
        vx = np.linspace(xmin, xmax, 100)

        w1 = model.weights[0,0] / scaler.std[0]
        w2 = model.weights[1,0] / scaler.std[1]

        b = model.bias - (
            model.weights[0,0]*scaler.mean[0]/scaler.std[0] +
            model.weights[1,0]*scaler.mean[1]/scaler.std[1]
        )
        
        if abs(w2) < 1e-10: return

        vy = -(b + w1*vx)/w2
        
        # Plot da fronteira
        plt.plot(vx, vy, color='red', label='Fronteira de decisão')

        for i in range(original.shape[0]):

            if y[i,0] == 1:
                plt.scatter(original[i,0], original[i,1], marker='o', color='blue')
            else:
                plt.scatter(original[i,0], original[i,1], marker='s', color='green')

        # limites dos eixos
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Organização do gráfico
        plt.xlabel('Entrada x1')
        plt.ylabel('Entrada x2')
        plt.title('Perceptron - Fronteira de decisão')
        plt.legend()
        plt.grid(True)

        plt.show()