# Homework 1

This repository contains the files for the first homework assignment.

## 📂 Directory Structure

Here's a quick look at the main files and folders in this repository:

```
.
├── data/
│   └── hmw1_25.txt         # Data file for analysis
├── presentation/
│   ├── figures/            # Folder to save plots and figures
│   └── notebooks/
│       ├── pregunta_1.ipynb    # Notebook for AR model analysis & IRF
│       ├── pregunta_2.ipynb    # Notebook for Monte Carlo simulations
│       └── pregunta_3.ipynb    # Notebook for ARMA model analysis
└── source/
├── models/
│   └── autoregressive.py # Main AR model class
└── utils/
├── plot.py           # Functions for plotting
└── random.py         # Functions for data generation
```

## 🚀 Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites

Make sure you have **Python 3** installed. If you don't, you can download it from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the repository** to your computer:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```

2.  **Navigate into the project directory**:
    ```bash
    cd your-repository-name
    ```

3.  **Install the required libraries**. This project uses a few Python libraries that you can install easily with pip:
    ```bash
    pip install numpy pandas matplotlib statsmodels jupyter
    ```

## 💻 How to Run the Code

The analysis in this project is done using Jupyter Notebooks. You can run them using Visual Studio Code (recommended) or directly in your browser.

### Option 1: Using Visual Studio Code (Recommended)

1.  **Open the project folder** in Visual Studio Code.
2.  **Install the Python and Jupyter extensions**. If you don't have them, VS Code will prompt you to install them when you open a notebook file.
3.  Use the **Explorer** panel on the left to navigate to the `presentation/notebooks/` directory.
4.  **Click on any `pregunta_X.ipynb` file** to open it.
5.  If prompted, select your Python interpreter (the one where you installed the libraries) by clicking in the top-right corner of the window.
6.  Run each block of code (called a "cell") by clicking the **Play** button next to it or by selecting the cell and pressing **Shift + Enter**.

### Option 2: Using Your Web Browser

1.  **Start the Jupyter Notebook server** by running this command in your terminal (from the project's root folder):
    ```bash
    jupyter notebook
    ```

2.  Your web browser will open a new tab. From there, navigate to the `presentation/notebooks/` directory.

3.  Click on any of the `pregunta_X.ipynb` files to open them and run the cells.

## 📝 Code Overview

This project is built around a few key files:

* **`source/models/autoregressive.py`**: This is the core of the project. It contains the `AR` class, which lets you define an Autoregressive model.

* **`source/utils/plot.py`**: This script helps create question figures.

* **`presentation/notebooks/`**: These notebooks put everything together. They show how to use the `AR` class and the utility functions to perform a complete analysis. If you're new to the code, these notebooks are the best place to start!