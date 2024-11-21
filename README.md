# Anomaly And Threat Detection In Network Traffic:

## About:

This project focuses on the identification of unusual patterns, anomalies, and potential security threats in network traffic data. Traditional security measures like **firewalls** and **intrusion detection systems (IDS)** struggle to keep up with increasingly complex cyberattacks and **zero-day exploits**. The goal of this project is to enhance cybersecurity by leveraging machine learning techniques to detect anomalies in static datasets first and then extend the model to predict potential threats in real-time network traffic.

By applying advanced machine learning algorithms, the project aims to improve the detection rate of various network-based attacks while minimizing false alarms. This system is intended to act as a supplementary layer of defense for existing security frameworks, detecting both known and previously unseen (zero-day) threats.

## Dataset:

The dataset used in this project is the **UNSW-NB15 dataset**, which was created by the Cyber Range Lab at UNSW Canberra. It includes network traffic data with a mix of normal activities and various types of cyberattacks.

- **Traffic Data**: The dataset contains a hybrid of real modern normal activities and synthetic contemporary attack behaviors.
- **Attack Types**: It includes 9 different attack categories:
  - Fuzzers
  - Analysis
  - Backdoors
  - DoS (Denial of Service)
  - Exploits
  - Generic attacks
  - Reconnaissance
  - Shellcode
  - Worms

The dataset includes 49 features that describe the network behavior of different traffic flows.

You can download the complete dataset from the official website here:
[The UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

### Key Features of the Dataset:

- **2.54 million records** distributed across multiple CSV files.
- **Training and Testing Sets**:
  - `UNSW_NB15_training-set.csv`: 175,341 records.
  - `UNSW_NB15_testing-set.csv`: 82,332 records.
- **Ground Truth**: The file `UNSW-NB15_GT.csv` contains the labels that indicate whether each record corresponds to normal traffic or an attack.

## Project Flow:

1. **Data Collection**:

   <!-- - Load the UNSW-NB15 dataset into your workspace.
   - Preprocess the data by handling missing values and normalizing the features. -->

2. **Exploratory Data Analysis (EDA)**:

   <!-- - Visualize and understand the distribution of normal vs attack data.
   - Identify any correlations between features and anomalies using data visualization tools. -->

3. **Feature Selection**:

   <!-- - Identify the most relevant features for anomaly detection.
   - Use techniques like **correlation analysis** and **feature importance** to select the best features for the model. -->

4. **Modeling**:

   <!-- - Train machine learning models (e.g., Logistic Regression, Random Forest, SVM) on the training dataset to classify network traffic as either normal or anomalous.
   - Explore both **supervised** and **unsupervised** learning techniques to detect anomalies.
   - Implement deep learning models (optional) for advanced threat detection. -->

5. **Model Evaluation**:

   <!-- - Test the model on the testing dataset to evaluate its performance.
   - Use evaluation metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score** to assess the detection capability of the model. -->

6. **Real-time Anomaly Detection** (Optional):
   <!-- - Extend the system to analyze real-time network traffic data.
   - Integrate the trained model into a real-time monitoring system to detect threats in live network traffic. -->

## Directory Structure For Project Setup:

<pre>
│   .gitignore
│   01_DatasetPreloading.ipynb
│   02_EDA.ipynb
│   03_DecisionTree.ipynb
│   04_kNN.ipynb
│   05_RandomForest.ipynb
│   LICENSE
│   README.md
│   
├───DATASET
│   ├───EDA-DATASET
│   │       x_test_scaled.csv
│   │       x_train_scaled.csv
│   │       y_test.csv
│   │       y_train.csv
│   │       
│   ├───FULL-DATASET
│   │       fullDF.csv
│   │       
│   └───ORIGINAL-DATASET
│       │   NUSW-NB15_features.csv
│       │   NUSW-NB15_GT.csv
│       │   The UNSW-NB15 description.pdf
│       │   UNSW-NB15_1.csv
│       │   UNSW-NB15_2.csv
│       │   UNSW-NB15_3.csv
│       │   UNSW-NB15_4.csv
│       │   UNSW-NB15_LIST_EVENTS.csv
│       │
│       └───TRAINING-AND-TESTING-SET
│               UNSW_NB15_testing-set.csv
│               UNSW_NB15_training-set.csv
</pre>

## Results:

- Once the model is trained, it should be able to detect anomalies in network traffic with high accuracy.
- The model will be evaluated based on metrics like Accuracy, Precision, Recall, and F1-Score, ensuring both known and unknown threats are identified efficiently.
