# OCD Detection from Multichannel EEG Signals

**6th Semester Minor Project**

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20Bi--LSTM-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression%20%7C%20Random%20Forest-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-91%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-Multichannel%20EEG-lightgrey)

## 📌 Project Overview
Obsessive-Compulsive Disorder (OCD) is a mental disorder marked by persistent thoughts and compulsive actions. This project demonstrates a Deep Learning-based framework to automate the detection of OCD vs. Non-OCD subjects using raw, multichannel Electroencephalogram (EEG) signals. 

By avoiding subjective, manual diagnoses, our approach leverages sequence models like **Long Short-Term Memory (LSTM)** and **Bidirectional LSTM (Bi-LSTM)** to capture complex temporal dependencies across brain regions, achieving significantly better performance than traditional machine learning methods.

**Reference Paper:** *LSTM and Bi-LSTM Based Deep Learning Approach to Detect Obsessive-Compulsive Disorder Using Multichannel EEG Signals*

## 🧠 Dataset & Preprocessing
The dataset consists of **50,000 EEG samples** with a focus on binary classification (OCD vs. Non-OCD).
* **Data Features:** 32 EEG channels, sampled at 256 Hz in the time domain, along with clinical data (sex, age, IQ, EQ).
* **Signal Processing:** Band-pass filtering (0-50 Hz) and signal normalization were applied to eliminate noise.
* **Data Augmentation:** To tackle class imbalance, we applied techniques such as adding Gaussian noise, amplitude scaling, and temporal shifting to amplify the number of OCD samples.

## 📁 Repository Structure
The core development and experiments were carried out in Jupyter notebooks (Google Colab).

* **`jupyter files/Aug_conversion.ipynb`**  
  Handles the data loading, pre-processing, one-hot encoding of clinical features, and data augmentation. It structures the 256 time-steps into a format suitable for initial modeling and scales the numerical features.

* **`jupyter files/training_testing.ipynb`**  
  Manages the train-test split (80:20) using stratification to maintain class distribution, preparing robust datasets for our classifiers.

* **`jupyter files/Logistic_Regression_and_Random_forest_aug_data_MAIN.ipynb`**  
  Implements traditional Machine Learning baselines. It extracts statistical features (mean, standard deviation, max, min, sum of squares) from the signal arrays and trains Logistic Regression and Random Forest classifiers.

* **`jupyter files/FINAL_LSTM_AUG_DATASET_MAIN.ipynb`**  
  Implements the standard Long Short-Term Memory (LSTM) network directly on the temporal sequences. Achieves ~88% accuracy, utilizing an Early Stopping callback to prevent overfitting.

* **`jupyter files/FINAL_BILSTM_AUG_DATASET_MAIN.ipynb`**  
  The main state-of-the-art approach for this project. Uses a Bidirectional LSTM to learn context from both the past and future sequences of the EEG signal. It successfully achieves the best results, reaching **91% accuracy**.

## 📊 Results & Performance
The deep learning models vastly outperformed traditional feature-engineered Machine Learning algorithms. The Bidirectional LSTM performed best due to its ability to capture both preceding and succeeding temporal contexts in the EEG signals.

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 50.0% | 0.52 | 0.50 | 0.50 | 0.49 |
| Random Forest | 66.0% | 0.76 | 0.66 | 0.58 | 0.63 |
| **LSTM** | **88.0%** | **0.88** | **0.88** | **0.88** | **0.94** |
| **Bi-LSTM** | **91.0%** | **0.91** | **0.91** | **0.91** | **0.96** |

### Visualizations
*(You can find detailed graphs for Accuracy vs. Epochs, Confusion Matrices, ROC Curves, and Precision-Recall Curves generated inside the respective deep learning notebooks).*

## 🚀 Setup and Installation
To run the notebooks locally or on Google Colab:
1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```
3. Load the dataset (or ensure the Colab environment has Google Drive mounted to access the preprocessed data).
4. Run the notebooks in the sequential order described in the repository structure.

## 👥 Contributors
This project was developed collaboratively by our team at MANIT, Bhopal for the 6th Semester Minor Project:
* **Harish Kushwaha**
* **Meet Lad**
* **Harish Patel**
* **Tripati Mittal**

