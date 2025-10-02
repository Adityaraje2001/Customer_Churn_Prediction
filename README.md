# 🧑‍💻 Customer Churn Prediction using Artificial Neural Networks (ANN)

## Project Overview

This repository contains the implementation of an Artificial Neural Network (ANN) model designed to predict customer churn for a bank. The goal is to identify customers who are likely to leave the bank (i.e., "exit") based on various features such as credit score, geography, gender, balance, and estimated salary.

The implementation is a practical exercise based on the Deep Learning community session by Krish Naik, focusing on key ANN concepts like feature scaling, categorical encoding, model compilation, and the use of the Adam optimizer and Early Stopping.

## Contents

* **`Customer_Churn_Prediction.ipynb`**: The main Google Colab notebook containing all the Python code for data preprocessing, model building, training, evaluation, and prediction.
* **`churn.modeling.csv`**: The dataset used for training and testing the ANN model (details below).
* **`README.md`**: This file.

## Technologies and Libraries

The project is built using the following core technologies and Python libraries:

* **Python**
* **TensorFlow/Keras**: For building and training the Artificial Neural Network.
* **Scikit-learn (sklearn)**: For data preprocessing (StandardScaler, Train-Test Split) and model evaluation (Confusion Matrix, Accuracy Score).
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Matplotlib**: For plotting training history (Loss and Accuracy).

## Dataset

The model is trained on the **`churn.modeling.csv`** dataset. It contains 10,000 customer records with 14 features and a binary target variable (`Exited`), where:
* **0** indicates the customer did **not** churn.
* **1** indicates the customer **did** churn.

## How to Run the Notebook

You can execute the entire project directly in Google Colab or on your local machine.

### Option 1: Run in Google Colab (Recommended)

1.  Click the following badge to open the notebook directly in Colab:
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yI6Zpb1lAvxVsXScQJDcD_MPh0O5_lt_#scrollTo=RzfZ3N2f_ySB)
2.  Once opened, ensure the dataset (`churn.modeling.csv`) is uploaded to the Colab environment (the notebook may contain code to handle this).
3.  Run the cells sequentially from top to bottom.

### Option 2: Clone and Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/adityaraje2001/Customer_Churn_Prediction.git](https://github.com/adityaraje2001/Customer_Churn_Prediction.git)
    cd Customer_Churn_Prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib
    ```
3.  Run the `Customer_Churn_Prediction.ipynb` notebook using a Jupyter environment.

## Model Summary

The Artificial Neural Network consists of an input layer, two hidden layers, and a final output layer, trained with the following configuration:

| Component | Setting |
| :--- | :--- |
| **Model Type** | Sequential ANN |
| **Optimizer** | Adam |
| **Loss Function** | Binary Cross-Entropy |
| **Metric** | Accuracy |
| **Regularization** | Early Stopping Callback |
