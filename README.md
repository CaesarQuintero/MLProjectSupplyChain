# MLProjectSupplyChain
Machine learning project for the detection and prediction of whether a load will arrive late or not using random forest
# Project Overview
>**Author's note:** I recommend reading this initial part to understand this model, this beta a full production model offers a complete model of anomalous case prediction, while reading the code it has instructions and notes that help a better interpretation.

**Developed by:** Cesar Augusto Quintero Guerra 2023

**Advised and reviewed by:** Johan Sanchez y Juan Lopez

**Lynxus 2023**


>Below you will find the project repository, along with all the development and planning of the project in notion including the production model.

* [Github Repository](https://github.com/CaesarQuintero/MLProjectSupplyChain)

* [Project in Notion](https://www.notion.so/Machine-Learning-Project-abc63e69e99643cb9eb3a51428deb061?pvs=4)

* [Production Model](https://colab.research.google.com/drive/14ulBobu4QZ5tPRMG2uqvU0nkn2i1uO8L?usp=sharing)

* [Linkedin](https://www.linkedin.com/in/caesarquintero/)


## Purpose

*   Create a machine learning model to predict freight delivery timeliness (on-time or late).
*   Employ Random Forest classifier with SMOTE oversampling and Random
* Undersampling to address class imbalance and enhance performance.

## Target Audience
* Individuals or organizations involved in freight transportation seeking to improve delivery reliability.

## Additional Details: Oversampling and Undersampling

### Oversampling:
Creates additional samples of the minority class (SMOTE used in this project).
### Undersampling:
Removes samples from the majority class (Random Undersampling used in this project).
<br/><br/>

---


# Installation Instructions
## Prerequisites
Python 3
## Installation

```
pip install pandas numpy matplotlib seaborn statsmodels gradio
```
<br/><br/>

---


## Usage Instructions
1.   **Load the dataset:**

```
data = pd.read_excel("MLDatasetforTest.xlsx", usecols=desired_columns)

```
**Note**: Loading Excel files can be slow, so it is recommended to use CSV files instead.

2.   **Preprocess the data (refer to code for detailed steps):**
  * Handle null and duplicate values
  * Remove outliers
  * Create binary target variables for late deliveries

3. **Undersample the majority class and oversample the minority class:**

#### Undersampling
```
rus = RandomUnderSampler(sampling_strategy='majority')
x_train_ontime, y_train_ontime = rus.fit_resample(x_train, y_train)
```
#### Oversampling
```
sm = SMOTE(sampling_strategy='minority',random_state=123)
x_train_late, y_train_late = sm.fit_resample(x_train, y_train)
```
4. **Split data into training and testing sets:**

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
```

5. **Combine oversampled and undersampled data:**


```
x_train_smote = pd.concat([x_train_late, x_train_ontime])
y_train_smote = pd.concat([y_train_late, y_train_ontime])
```

6. **Create and train the Random Forest classifier:**

```
random_forest_model_SMOTE = RandomForestClassifier(**hyperparameters)
random_forest_model_SMOTE.fit(x_train_smote, y_train_smote)
```

7. **Make predictions:**

```
predictions = random_forest_model_SMOTE.predict(x_test)
```

8. **Evaluate model performance:**

```
print(classification_report(y_test, predictions))
print('Accuracy:', accuracy_score(y_test, predictions))
```

9. **Deploy the model:**

* Command-line interface (1st Deployment Option)
* Gradio web interface (2nd Deployment Option)


## Additional Information

##**Feature Importance**
### Most influential features:
* FreightWeight
* Miles
* CustomerCharges

##**Hyperparameter Tuning**
###Tuned hyperparameters:
* n_estimators = 60
* max_depth = 4

## Troubleshooting

Missing libraries: pip install missing libraries.
Data format issues: Ensure dataset is in Excel format with specified columns.
Model errors: Double-check code syntax and hyperparameter values.
