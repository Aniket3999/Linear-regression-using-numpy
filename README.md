# Linear Regression from Scratch with NumPy

This project provides a fundamental, step-by-step implementation of Polynomial Linear Regression using only the **NumPy** library. The Jupyter Notebook is designed as an educational tool, breaking down the core mathematical concepts—such as the cost function and gradient descent—and translating them directly into code.

It's a great resource for anyone looking to understand the mechanics behind one of the most fundamental machine learning algorithms without relying on high-level libraries like Scikit-learn.

## 📈 Demonstration

The model successfully learns the parameters to fit a quadratic curve to synthetically generated data. The final output visualizes the learned model against the original data points.

![Polynomial Regression Fit](https://i.imgur.com/gKzN5j7.png)
*(You can generate this plot by running the final cell in the notebook.)*

---

## ✨ Key Features

* **Pure NumPy Implementation:** The entire algorithm is built from the ground up using only NumPy for numerical operations.
* **Batch Gradient Descent:** Demonstrates the implementation of Batch Gradient Descent to optimize the model's parameters.
* **Polynomial Regression:** Shows how to extend linear regression to fit non-linear data by creating polynomial features.
* **In-depth Explanations:** The notebook is rich with comments and explanations that connect the mathematical theory to the code.
* **Data Visualization:** Uses Matplotlib to visualize the initial dataset and the final regression curve, providing a clear view of the model's performance.

---

## 📚 Core Concepts Covered

This notebook is a practical guide to the following concepts:

* **Hypothesis Function:** The linear model's prediction function.
    $$h_{\theta}(x) = \theta^T \cdot x$$
* **Cost Function (Mean Squared Error):** The function used to measure the model's performance.
    $$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$
* **Batch Gradient Descent:** The optimization algorithm used to find the best model parameters ($\theta$) by minimizing the cost function.
    $$\theta_j := \theta_j + \alpha \frac{2}{m} \sum_{i=1}^{m} (y^{(i)} - h_{\theta}(x^{(i)}))x_j^{(i)}$$
    *(Note: The notebook implementation uses a slightly different but equivalent update rule form: `theta = theta + learning_rate * batch_gradient.T`)*
* **Vectorization:** Using NumPy to perform matrix operations efficiently, which is crucial for performance in machine learning.
* **Feature Engineering:** Creating polynomial features to allow a linear model to fit non-linear data.

---

## 🛠️ Technologies Used

* **Python 3.x**
* **NumPy:** For all numerical and matrix operations.
* **Matplotlib:** For data visualization.
* **Jupyter Notebook:** As the interactive environment for code and explanations.
