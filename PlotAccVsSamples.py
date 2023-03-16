import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_final_accuracy_vs_samples(directory):
    num_samples = []
    final_accuracies = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as file:
                data = json.load(file)
                for key in data:
                    epochs = data[key]["epochs"]
                    _, accuracy = zip(*epochs)
                    samples = data[key]["num_samples"]
                    num_samples.append(samples)
                    final_accuracies.append(accuracy[-1])

    # Plot final accuracy vs num_samples
    plt.figure()
    plt.scatter(num_samples, final_accuracies, label='Data points')

    # Fit a linear regression line
    coefficients = np.polyfit(num_samples, final_accuracies, 1)
    poly = np.poly1d(coefficients)
    x = np.linspace(min(num_samples), max(num_samples), 100)
    plt.plot(x, poly(x), 'r-', label='Best fit line')

    # Calculate R-squared value
    r_squared = np.corrcoef(num_samples, final_accuracies)[0, 1] ** 2
    plt.xlabel('Number of Samples')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy vs Number of Samples CNN to Transformer')
    plt.legend(title=f'R^2: {r_squared:.3f}')
    plt.show()


def main():
    plot_final_accuracy_vs_samples('results/resultsCNNTransformer')


if __name__ == '__main__':
    main()
