# Introduction

**Analyzing the Stellar Classification Dataset from SDSS17** 

In this report, we delve into an intriguing dataset from Kaggle: the Stellar Classification Dataset - SDSS17, meticulously compiled by fedesoriano in January 2022. The dataset, accessible at [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data), presents a rich tapestry of astronomical data pivotal for the classification of stars, galaxies, and quasars. Such classifications, rooted in spectral characteristics, form a cornerstone in the expansive field of astronomy. The historical progression from early star cataloguing to the modern distinction between various celestial bodies underscores the importance of this endeavor.

The dataset encompasses 100,000 observations from the Sloan Digital Sky Survey (SDSS), each marked by 17 feature columns and a class column. These features include crucial identifiers like object ID, right ascension, declination angles, and photometric system filters (ultraviolet, green, red, near-infrared, and infrared). Additional data points such as run ID, rerun ID, camera column, field ID, unique IDs for optical spectroscopic objects, redshift value, plate ID, Modified Julian Date, and fiber ID enhance the dataset's depth, offering a comprehensive view of each celestial object as either a star, galaxy, or quasar.

This project aims to analyze this dataset and contrast the findings with those of fellow Kaggle contributors. A key hypothesis underpinning this endeavor is the assertion that many Kaggle users may not fully leverage the necessary data processing methods before integrating the data into their models. The belief is that thorough and appropriate data processing is crucial for extracting meaningful insights. By implementing meticulous data processing techniques, it is anticipated that one can achieve greater accuracy and a higher Receiver Operating Characteristic (ROC) Area Under the Curve (AUC) in the classification of these celestial entities.

Through this analytical journey, we aim to shed light on the nuances of data processing in astronomical datasets and its impact on model performance, hoping to contribute to the broader understanding of stellar classification.

---
### Contributors
This project was a collaborative effort and benefited immensely from the contributions of Thirukumaran Velayudhan and Elian Morales Pina. Their expertise in Machine Learning and Data Exploratory Analysis techiques was invaluable in the analysis and interpretation of the dataset.

---
### Important Links 
- **[Video: Presentation of the Stellar CLlassification DAE](/)**
- **[GitHub Repository](/)**
- **Thirukumaran Velayudhan**
  - [Milestone 2 Proposal](/)
- **Elian Morales Pina**
  -  [Milestone 2 Proposal](/)

---
### Report

#### **Methodology** 
For the core of our project, we chose Python as our programming language. Python is widely recognized for its versatility and the extensive support it provides for data analysis and machine learning tasks.

In terms of data manipulation, we utilized two key Python libraries: NumPy and Pandas. For visualizing our data, we turned to Matplotlib and Seaborn. Lastly, for all tasks related to machine learning, we employed Scikit-learn (commonly known as sklearn). This library is a powerful tool for machine learning in Python, offering a range of algorithms for classification, regression, clustering, and more.

**Initial Data Splitting:** First, we split original dataset (stellar_data) into a train_df (df) and a test_df. The test_df is set aside and not used for anything until the very end of the process.
- **Note**: We rename the train_df as df

**Exploratory Data Analysis (EDA):** Perform EDA on the train_df (df) only. This is because EDA can inform feature engineering and preprocessing steps, and doing this on the entire dataset could lead to data leakage where information from the test set could influence the model training process.

**Further Split for Validation:** To build a model, we took the train_df (df) and split it again into training and validation datasets. This new validation set (val_df) is used to simulate the test set during the model selection and tuning process.

**Model Training and Selection:** Use the training portion of train_df (df) to train your models and the validation set to tune hyperparameters and make decisions about which models and features work best.

**Final Test:** After selected the best model and trained it on the full train_df (df: which includes the validation data), we finally evaluate it on the test_df to measure the model's accuracy and performance. This is your estimate of how the model will perform on unseen data.

**Data Exploratory Analysis:**
Following the initial data splitting, we moved on to a more in-depth phase of our Data Exploratory Analysis (EDA). This involved a thorough examination of each attribute within our dataset. Here’s how we approached it:
- **Attribute Analysis:** We studied each attribute in detail, focusing on its name and type. The types we looked at included categorical, integer or float (numerical), bounded or unbounded values, text, and structured formats. This helped us understand the nature of the data we were dealing with.
- **Dealing with Missing Data:** We paid special attention to missing data in our dataset. Identifying where and why data was missing was crucial for ensuring the quality of our analysis.
- **Evaluating Noisiness and Distributions:** We assessed the noisiness in the data and the type of distributions each attribute had. Understanding these aspects was essential for later stages of data processing and model selection.
- **Identifying the Target Attribute:** Since our project was centered around a supervised learning task, it was vital to identify the target attribute – the variable that we aimed to predict with our models.
- **Data Visualization:** We created various visualizations of the data. These visualizations helped us to better understand the relationships within the data and to communicate our findings effectively.
- **Correlation Study:** An important part of our analysis was studying the correlations between different attributes. This helped us understand how different variables related to each other and their potential impact on our target attribute.
- **Exploring Promising Transformations:** We explored various transformations that could be applied to our data to make it more suitable for modeling. This step was about finding ways to enhance the data’s characteristics to improve our model’s performance.
- **Documentation of Findings:** Throughout our exploratory process, we meticulously documented all our findings and observations. This was done in our Jupyter Notebook, which served as a comprehensive record of our analysis journey. This documentation not only helped in keeping our analysis organized but also ensured that our process was transparent and reproducible.

In summary, this phase of our methodology was dedicated to gaining a deep understanding of our dataset’s characteristics and preparing it for the subsequent modeling stages.

**Data Processing:**
In our data preparation phase, we prioritized maintaining the integrity of the original dataset. To achieve this, we worked exclusively with copies of the data. This approach ensured that the original dataset remained unaltered throughout our analysis.

**Automated Data Transformation:** To streamline our data preparation process, we developed functions for all the data transformations we planned to apply. This strategy allowed us to efficiently prepare both the training and test data. Additionally, it enabled us to treat our preparation choices as hyperparameters, providing flexibility in our approach and facilitating easy adjustments.

**Data Cleaning**
Handling Outliers: We employed three different techniques to manage outliers:
- The Local Outlier Factor (LOF) algorithm from Scikit-learn.
- Identifying outliers using the Interquartile Range (IQR).
- Applying the Z-score method.
  
**Managing Missing Values:** For handling missing values, we considered three approaches:
- Dropping rows or columns with null values.
- Replacing null values with the median of the respective column.
- Substituting null values with the mean of the column.

**Feature Selection:** Our approach to feature selection involved several steps:
1. Examining the correlations between features, and their correlation with the target column.
2. Creating visualizations of attribute interactions, using the target class as a hue, to deepen our analysis.
3. Utilizing a Random Forest Classifier to determine feature importances. Based on these importances, we finalized our feature selection.

**Feature Engineering**
In this stage, we applied a label encoder to categorical data. We also explored various transformations, such as logarithmic (log(x)), quadratic (x^2), and square root transformations, which showed promise in enhancing our model’s performance.

**Addressing Class Imbalance**
During our Data Exploratory Analysis (DAE), we identified an imbalance in our class labels. To address this, we implemented rebalancing techniques, focusing on oversampling the minority classes ('STAR' and 'QSO'). This was achieved by replicating existing examples or generating new ones using techniques like Synthetic Minority Over-sampling Technique (SMOTE), which helped to equalize the representation of different classes in our target column.

**Feature Scaling**
Finally, we conducted feature scaling by standardizing our features. This process involved adjusting the values of each feature in our dataset to have a mean of zero and a standard deviation of one, ensuring that all features contributed equally to the model’s performance.

In summary, our data preparation phase was comprehensive, involving multiple steps from cleaning and feature selection to engineering and scaling. Each step was carefully planned and executed to optimize our dataset for the machine learning models.

#### **Methodology: Model Training and Selection**
**Training Multiple Models**
Once our data was fully prepared, we moved on to the next critical phase: training various models. Our strategy here was to start with a diverse set of 'quick and dirty' models. This included a range of model types such as linear models, Naive Bayes, Support Vector Machines (SVM), Random Forests, and even neural networks. We kept the parameters for these models at their standard settings, aiming to get a baseline performance for each.

**Cross-Validation and Performance Assessment**
For each of these models, we implemented N-fold cross-validation. This technique involves splitting the training data into N parts, training the model N times, each time using a different part as the test set and the remaining as the training set. After running these cycles, we calculated the mean and standard deviation of the model's performance across the N folds. This approach provided a robust assessment of each model's performance and reliability.

**Analyzing Model Errors and Selecting Promising Models**
A crucial part of our process was to analyze the types of errors made by each model. Understanding these errors gave us insights into the strengths and weaknesses of the models in handling our data. Based on this analysis, we selected the top three most promising models. Our selection criteria were not just based on overall performance, but also on the diversity of errors. We preferred models that made different types of errors, as this could be advantageous in developing a more robust final model.

**Implementing Grid Search for Data Processing**
In our pursuit to optimize the data processing pipeline, we employed grid search as a key technique. This approach involved treating each data processing method as a hyperparameter. By doing so, we systematically worked through multiple combinations of these methods to determine the most effective data processing strategy. This grid search not only fine-tuned our models but also enhanced the overall quality of the data feeding into them.

**Fine-Tuning Top Models with Cross-Validation**
After identifying our top three machine learning models based on their initial performance and error analysis, we moved on to fine-tune these models. We used cross-validation for this purpose, ensuring that our fine-tuning was thorough and avoided overfitting. This step was crucial in maximizing the potential of each selected model.

**Implementing a Voting Classifier**
Another innovative step in our methodology was the implementation of a voting classifier. This classifier combined our top three machine learning models. The idea behind this was to leverage the strengths of each individual model. By doing so, we aimed to create a more robust and accurate prediction system, capitalizing on the diverse capabilities of the chosen models.

**Random Search for Hyperparameter Optimization**
Given the vast amount of data and the extensive range of hyperparameters involved, we opted for a random search instead of a grid search for hyperparameter optimization. Random search allowed us to explore a wide range of parameters more efficiently, making it a more practical choice in our context where computational resources and time were factors to consider.

**Performance Measurement of the Models**
Finally, we reached the stage of measuring the performance of our top three classification models and the voting classifier. To assess their effectiveness, we used several metrics:
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** This metric helped us understand the models' ability to distinguish between classes.
- **Accuracy:** This provided a straightforward measure of how often the models correctly classified instances.
Classification Report (classification_report(y_test, y_pred)): This report gave us detailed insights into the precision, recall, and F1-score for each class.
- **Confusion Matrix (confusion_matrix(y_test, y_pred)):** This matrix was useful in visualizing the performance of the models in terms of correctly and incorrectly classified instances.







