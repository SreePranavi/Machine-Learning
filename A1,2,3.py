#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# To Load the excel data
def load_data(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


# Function to calculate dimensionality
def dimensions(A):
     # shape is an attribute that provides the dimensions of an array present in numpy library
     # it returns a tuple representing the dimensions
    dimensionality = A.shape
    print(f"Dimensionality of the vector space: {dimensionality}")

   
# Function to count the number of vectors
def count_vectors(A):
     # A.shape[0] gives the number of rows
        #A.shape[1] gives the number of columns
    num_vectors = A.shape[0]
    print(f"Number of vectors in this vector space: {num_vectors}")

       
# Function to find the rank of Matrix A
def rank_of_matrix(A):
    # numpy.linalg.matrix_rank() gives the number of linearly independent rows or columns
    rank_A = numpy.linalg.matrix_rank(A)
    print(f"Rank of Matrix A: {rank_A}")

# Function to compute pseudo-inverse and find costs
def compute_costs(A, C):
    # np.linalg.pinv() is used to get the inverse of A
    A_pseudo_inv = numpy.linalg.pinv(A)
   
    # X is used to get the cost of the pseudo inverse
    #dot() performs dot product
    # this function call performs the matrix multiplication
    X = np.dot(A_pseudo_inv, C)
    return X


# Function to mark customers as RICH or POOR and build a classifier
def classify_customers(df):
    # Mark customers
    df['Class'] = np.where(df['Payment (Rs)'] > 200, 'RICH', 'POOR')
   
    # Prepare features and labels
    X = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    y = df['Class'].values
   
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
   
    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)
   
    # Predict on the test set
    y_pred = model.predict(X_test)
   
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Main function to execute the tasks
def main():
    # Load the data
    file_path = "C:/Users/year3/Downloads/Lab Session Data (1).xlsx"
    sheet_name = 'Purchase data'
    df = load_data(file_path, sheet_name)

    # Selection of specific columns
    columns_to_print = df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]
    print("Selected columns:")
    print(columns_to_print)
   
    # Create matrices A and C
    A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
    C = df[['Payment (Rs)']].values
   
    # Print matrices
    print("\nMatrix A:")
    print(A)
    print("\nMatrix C:")
    print(C)
   
    # Perform calculations
    dimensions(A)
    count_vectors(A)
    rank_of_matrix(A)
    X = compute_costs(A, C)
    print("\nCost of each product available for sale:")
    print(X)
   
    # Classify customers
    classify_customers(df)

# Execute the main function
if __name__ == "__main__":
    main()


# In[ ]:




