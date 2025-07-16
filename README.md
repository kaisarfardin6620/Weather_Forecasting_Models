# Weather_Forecasting_Models

Weather Analysis and Machine Learning Models
This repository contains a Jupyter Notebook (Weather_Analysis_and_Modeling.ipynb) that performs a comprehensive analysis of historical weather data. The project covers data loading, extensive exploratory data analysis (EDA), robust data preprocessing, feature engineering, and the application and evaluation of various traditional machine learning classification models to predict daily weather summaries.

Table of Contents
Project Overview

Dataset

Notebook Structure

Features

Requirements

Installation and Setup

Usage

Results and Insights

Contributing

License

1. Project Overview
The primary objective of this project is to build and evaluate machine learning models capable of classifying daily weather summaries based on historical weather observations. The notebook walks through the entire data science pipeline:

Data Acquisition & Initial Inspection: Loading the dataset and understanding its basic characteristics.

Exploratory Data Analysis (EDA): Visualizing distributions, relationships, and trends within the data.

Data Preprocessing: Cleaning the data by handling missing values, duplicates, and outliers.

Feature Engineering: Creating new, informative features from existing ones to enhance model performance.

Model Training & Evaluation: Implementing and comparing several classification algorithms to determine their effectiveness.

2. Dataset
The project utilizes the weatherHistory.csv dataset. This dataset contains a rich collection of historical weather data, including:

Formatted Date

Summary

Precip Type

Temperature (C)

Apparent Temperature (C)

Humidity

Wind Speed (km/h)

Wind Bearing (degrees)

Visibility (km)

Loud Cover

Pressure (millibars)

Daily Summary (Target Variable)

3. Notebook Structure
The Weather_Analysis_and_Modeling.ipynb notebook is organized into the following sections:

1. Setup and Imports: Imports all necessary Python libraries and includes code for mounting Google Drive (if running in Colab).

2. Data Loading and Initial EDA: Loads the weatherHistory.csv file, displays DataFrame information, shape, columns, descriptive statistics, and checks for missing values and duplicates.

3. Visualizations for EDA: Generates various plots like histograms, box plots, correlation heatmaps, scatter plots, time series plots, bar plots, and pair plots to visually explore the data.

4. Feature Engineering: Creates new features such as Year, DayOfWeek, Is_Weekend, cyclical Hour_sin/Hour_cos, Month_sin/Month_cos, Humidity_Level, and several interaction features.

5. Outlier Handling and Encoding Categorical Features: Implements IQR-based outlier removal and uses LabelEncoder to convert categorical features into numerical format.

6. Data Splitting and Scaling: Divides the data into training and testing sets and applies StandardScaler to numerical features.

7. Model Evaluation Function: Defines a helper function to calculate and display common classification metrics (Accuracy, Recall, Precision, F1 Score, Classification Report).

8. Traditional Machine Learning Models (Before Tuning): Trains and evaluates a suite of scikit-learn classification models: Logistic Regression, Decision Tree, K-Nearest Neighbors, Random Forest, Support Vector Classifier (SVC), and Gaussian Naive Bayes.

9. Model Performance Comparison Visualization: Presents a bar chart comparing the performance metrics of all trained models.

4. Features
Comprehensive EDA: In-depth data exploration using various statistical summaries and visualizations.

Robust Preprocessing: Effective handling of missing data, duplicates, and outliers.

Advanced Feature Engineering: Creation of temporal and interaction features to capture complex patterns.

Multiple Model Evaluation: Comparison of several widely used classification algorithms to identify the best performers.

Clear Code Structure: Organized into logical sections within a Jupyter Notebook for easy understanding and execution.

5. Requirements
To run this notebook, you will need Python 3.x and the following libraries:

numpy

pandas

seaborn

matplotlib

scikit-learn

scipy

You can install these dependencies using pip:

pip install numpy pandas seaborn matplotlib scikit-learn scipy

6. Installation and Setup
Clone the repository:

git clone https://github.com/kaisarfardin6620/Weather_Forecasting_Models
cd YourRepoName


Download the dataset:
Ensure the weatherHistory.csv file is placed in a location accessible by the notebook. If you are running this in Google Colab, the notebook expects the file at /content/drive/MyDrive/dataset/weatherHistory.csv. If running locally, adjust the path in the pd.read_csv() line accordingly.

Open the notebook:
You can open and run the Weather_Analysis_and_Modeling.ipynb file using Jupyter Notebook, JupyterLab, or Google Colab.

For Google Colab: Upload the .ipynb file to your Google Drive and open it with Google Colab. Ensure your dataset is also in your Google Drive at the specified path or upload it to the Colab environment.

7. Usage
To execute the analysis and model training:

Open the Weather_Analysis_and_Modeling.ipynb notebook in your preferred environment (Jupyter, JupyterLab, or Google Colab).

Run all cells sequentially. The notebook is designed to be executed step-by-step, with outputs and visualizations generated along the way.

8. Results and Insights
The notebook will output various data descriptions and plots during the EDA phase. In the model evaluation section, it will print detailed classification reports and a comparative bar chart summarizing the performance (Accuracy, Recall, Precision, F1 Score) of each trained model. This allows for direct comparison and identification of the most suitable model for predicting daily weather summaries.

9. Contributing
Feel free to fork this repository, open issues, or submit pull requests. Any contributions to improve the analysis, add more models, or enhance visualizations are welcome!
