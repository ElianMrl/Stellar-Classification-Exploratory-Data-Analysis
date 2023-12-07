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
- **[Video: Presentation of the Stellar CLlassification DAE](https://urldefense.com/v3/__https://njit.webex.com/njit/ldr.php?RCID=7da11980a203273090524b2af8c0cc09__;!!DLa72PTfQgg!KI50kkLgmqFfb6mnd-WuPpJQYp4Yegx5Pm7KS0tjNaQL4lJ60rK7Nqxm0jf2q2NS98xwYzh8VFfITPED_dU$)**
- **[GitHub Repository](https://github.com/ElianMrl/Stellar-Classification-Exploratory-Data-Analysis/tree/main)**
- **Thirukumaran Velayudhan**
  - [Milestone 2 Proposal](https://github.com/ElianMrl/Stellar-Classification-Exploratory-Data-Analysis/blob/main/documents/Milestone%202%20-%20Thirukumaran%20Velayudhan.pdf)
- **Elian Morales Pina**
  -  [Milestone 2 Proposal](https://github.com/ElianMrl/Stellar-Classification-Exploratory-Data-Analysis/blob/main/documents/Milestone%202%20-%20Elian%20Morales%20Pina.pdf)

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

---
### **Results**

![img1](img/img1.png)

In our analysis, we observed that certain features exhibited a Right Skewed Distribution. Specifically, 'field_ID' and 'redshift' fell into this category.
- Log Transformation: We found that log transformation was effective for right-skewed data. However, it's important to note that this technique only works for positive values. Therefore, we ensured that the values, especially the delta values, were shifted to be positive before applying this transformation.
- Handling 'field_ID': Despite 'field_ID' being right-skewed, we decided not to perform any transformations on it. Our analysis suggested that 'field_ID' might be a categorical value, and transforming it could distort its categorical nature and significance.

![img2](img/img2.png)

Our data showed a clear imbalance among the classes. Galaxies were more frequent than both stars and QSOs. QSOs appeared less frequently compared to galaxies and stars, aligning with the expectation that QSOs are rarer and more distant celestial objects.

![img3](img/img3.png)

When examining the correlation between the target column ('class') and other features, we found that 'u' and 'g' had the highest correlation values with 'class':
- 'u': 0.27
- 'g': 0.23

![img4](img/img4.png)

Our investigation into the relationship of the 'u' feature with other features revealed challenges in classifying the class of the records. However, a notable exception was observed when comparing 'u' with 'redshift'.

![img5](img/img5.png)

Based on our findings, we decided to further explore 'redshift' against other features. This led to an important discovery: 'redshift' emerged as the most significant feature for classifying records as 'star', 'galaxy', or 'qso'.

![img6](img/img6.png)

As part of our analysis, we employed the RandomForestClassifier, a robust machine learning algorithm, to assess the importance of different features in classifying the records in our dataset. This method provided us with an importance score for each feature, enabling us to understand their relative significance in the classification process.
- The results from the RandomForestClassifier were quite revealing. They confirmed our previous observations regarding the importance of the 'redshift' feature. In fact, 'redshift' emerged as the most crucial feature for classification, standing out significantly among the others.

![img7](img/img7.png)

In our efforts to improve the distribution of the non-categorical columns, we experimented with various transformations such as log(X), square, and square root. However, these transformations did not yield the improvements we were hoping for in terms of achieving a more normalized distribution. The transformed distributions of these columns remained largely unchanged, indicating that these methods were not effective for our particular dataset.
- Interestingly, we observed that the columns 'Plate' and 'spec_obj_ID', which are categorical in nature, did show a bell-shaped distribution after applying transformations. However, given their categorical status, applying such numerical transformations is not appropriate. These transformations could distort the inherent categorical information and lead to misleading results.
- On a positive note, the distribution of the 'redshift' column showed some improvement after applying a square root transformation. This transformation made the distribution of 'redshift' slightly more normalized, which could potentially enhance its utility in our classification models.

**Balancing the Dataset with SMOTE**

![img8](img/img8.png)

We then addressed the issue of class imbalance in our dataset by using the Synthetic Minority Over-sampling Technique (SMOTE). This approach successfully balanced the dataset by equalizing the number of labels across different classes. As a result, each class (star, galaxy, QSO) was equally represented in our records.

While SMOTE effectively resolved the imbalance, it introduced two potential challenges:

- **Risk of Overfitting:** The artificial increase in data points for minority classes might lead to models that are too specifically tailored to the training data, reducing their generalizability.
- **Increased Computational Time:** The enlarged dataset resulted in longer computation times when fitting models, as there was more data to process.

![img9](img/img9.png)

**After running some quick and dirty models with the default hyperparameters we got the following Evaluation of the Models:**
**Random Forest:**
- **Mean Score:** 0.9763
- **Performance:** High precision and recall across all classes.
- **Type of Errors:** Very few errors, well-balanced across classes.
- Best overall performance with the highest mean score and balanced error distribution.

**MLP Classifier:**
- **Mean Score:** 0.9682
- **Performance:** High precision and recall, slightly lower than Random Forest.
- **Type of Errors:** Similar to Random Forest, but with slightly more errors in classifying class 0 and 1.

**Gradient Boosting:**
- **Mean Score:** 0.9659
- **Performance:** Good precision and recall, but not as high as Random Forest and MLP.
- **Type of Errors:** More errors in classifying class 0 and 1 compared to Random Forest and MLP.

After running the pipeline testing different data processing methods (using a grid search like algorithm) we ended up with the conclusion that the best processing method was:
- applying Random Forest with threshold = 0.007 for feature selection
- No column transformation
- Oulier Filter/Detection Method: 'LOF' with neighbors = 20 and threshold = -1.5
- For Dealing with null values, the best method is the replacing Null values with hte median
- Rebalancing the data uisng SMOTE

---
### Results: Model Performance and Overfitting Issue

**Outcome of Fine-Tuning Models**

After a rigorous process of fine-tuning our machine learning models and evaluating their performance using the test data, we concluded that the models were overfitting.

**Performance of MLPClassifier**

The Multi-Layer Perceptron (MLP) Classifier emerged as the standout performer among the models we fine-tuned. Its performance metrics were notably impressive:
- **Accuracy:** The MLPClassifier achieved an accuracy of 91.14%. This high accuracy indicates that the model was able to correctly classify a significant majority of the instances in the test dataset.
- **ROC-AUC Score:** The model scored 98.75% in the ROC-AUC metric. This high score reflects the model's strong capability in distinguishing between different classes, a crucial aspect in classification problems.

![img10](img/img10.png)

**Overfitting in Other Models**

On the other hand, we observed a concerning trend in the rest of the models. Despite showing promising results on the training data, their performance dropped significantly when applied to the test data. This discrepancy is a classic indication of overfitting.

#### Final Results: Adjustments and Model Performance

**Rationale for Data Re-Split**

![img11](img/img11.png)

In response to the overfitting issue observed in our initial model evaluations, we decided to take a step back and re-split the data. This decision was driven by the need to prevent data snooping, a scenario where the model inadvertently learns specific characteristics of the test data, thus compromising its ability to generalize. The new split allowed us to re-tune our models and adjust our data processing methods with fresh, unseen data, ensuring a more robust evaluation of model performance.

**Final Performance Metrics**

After making these adjustments, we conducted another round of evaluations on the models. Here are the key outcomes:

- **Random Forest Classifier:** Although there was a slight improvement in the Random Forest model's performance, it still struggled with accuracy. This indicated that despite fine-tuning, the model was unable to effectively generalize to the new test data.
- **Gradient Boosting Classifier:** Similar to the Random Forest, the Gradient Boosting Classifier also exhibited subpar accuracy. This was another indication that, like the Random Forest model, it was not adequately equipped to handle the complexity or specific characteristics of the test dataset.

![img12](img/img12.png)

- **MLPClassifier:** The standout performer in our final evaluation was the MLPClassifier. It achieved an accuracy of 94.87% and an impressive ROC-AUC score of 99.41%. These metrics indicated a significant improvement in the model's ability to correctly classify instances and effectively distinguish between classes. The high ROC-AUC score, in particular, suggested that the model had a strong capability in handling both classes proportionately well.

---
### Conclusion:

As a part of our comprehensive analysis, we decided to compare our approach with those of other Kaggle users who had also worked on this dataset. The primary objective of this comparison was to evaluate the effectiveness of our intensive data processing methods against more straightforward approaches used by others.

**Findings from the Comparison**
- **Performance Metrics:** Upon implementing the code from several Kaggle users, who employed relatively minimal data processing methods, we observed a striking result. The accuracy and ROC-AUC scores achieved using these simpler methods were surprisingly similar to those obtained from our more complex and intensive processing approach.

- **Implications on Data Processing:** This comparison led to a significant revelation about the dataset in question. It indicated that for this specific dataset, employing extremely detailed and intensive data processing methods did not result in a substantial improvement in the performance of the machine learning models. Instead, it primarily increased the computational time required for the Data Exploratory Analysis.

- **Efficiency vs. Effectiveness:** The similar performance outcomes suggest that a more streamlined, less computationally intensive approach could be equally effective for this type of data. This insight is particularly valuable in contexts where computational resources are limited or when rapid analysis is required.













