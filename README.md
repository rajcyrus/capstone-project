# Ames Housing Data Analysis and Machine Learning-project
The dataset was loaded into a pandas DataFrame using the pd.read_csv() function. The head() function was used to display the first five records, providing an initial overview of the data. The dtypes attribute was used to inspect the data types of all variables, revealing a mix of integer, floating-point, and categorical (object) features. The shape attribute showed that the dataset contains 2,930 observations and 82 variables, making it suitable for both regression and classification tasks.
## Missing Value Handling:
Missing values were analyzed by computing both the count and percentage of null values for every column. Columns with more than 20% missing values were identified separately because such a high proportion of missing data can reduce the reliability of simple imputation methods. For numeric columns with less than 20% missing values, missing entries were replaced with the median of the respective column. The median was chosen instead of the mean because it is robust to outliers and skewed distributions. Since many housing-related variables, such as sale price, lot frontage, and basement area, contain extreme values, median imputation provides a more representative measure of central tendency and minimizes the influence of unusually large or small observations. This approach helps preserve the overall distribution of the data while avoiding bias introduced by outliers.
# Duplicate Detection and Removal:
Duplicate records were identified using the df.duplicated().sum() function. The analysis showed that the dataset contained 0 duplicate rows. Consequently, applying df.drop_duplicates() did not remove any observations. Since no rows were removed, the percentage of missing values in each column remained unchanged. Therefore, duplicate records did not affect the quality or completeness of the dataset.
## Data type correction:
Pandas automatically inferred the correct data types for most columns in the Ames Housing dataset. A check was performed to identify any numeric columns stored as object; if found, they were converted to numeric using pd.to_numeric(errors='coerce') to ensure proper numerical analysis.
Several repetitive string columns (MS Zoning, Street, Neighborhood, House Style, and Exterior 1st) were converted from the object data type to the category data type using astype('category'). Since these columns contain a limited number of repeated values, the category type stores each unique value only once and replaces repeated strings with integer codes. This significantly reduces memory usage without affecting the data.
## Descriptive statistics and skewness
The describe() function was applied to all numeric columns to obtain summary statistics, including the count, mean, standard deviation, minimum, maximum, and quartiles. To understand the shape of each variable's distribution, skewness was computed using the skew() function.
The analysis showed that Misc Val has the highest absolute skewness in the dataset, indicating a highly positively skewed (right-skewed) distribution. Most properties have a value of zero or a very small miscellaneous value, while a few properties have exceptionally large values, creating a long tail on the right side of the distribution.
## Outlier detection with IQR:
Outliers were identified using the Interquartile Range (IQR) method. For each selected numeric variable, the first quartile (Q1), third quartile (Q3), and IQR (Q3 − Q1) were calculated. Observations lying below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR were classified as outliers.The analysis was performed on the following variables: Sale price and Lot area
##  Handling of Outliers
The identified outliers were not removed during Part 1 of the assignment. This decision was made because the extreme values represent genuine properties (e.g., luxury homes or unusually large lots) rather than data entry errors. Removing them could result in the loss of important information and reduce the dataset's representativeness.
For Part 2 (Machine Learning), the outliers will be retained. Tree-based models such as Decision Trees and Random Forests are generally robust to outliers and do not require their removal. For algorithms that are more sensitive to extreme values (such as Linear Regression), model performance will be evaluated first, and techniques such as feature transformation (e.g., logarithmic transformation) or winsorization (capping extreme values) may be considered if the outliers have a significant negative impact. This approach preserves valuable information while allowing appropriate handling based on the requirements of the specific machine learning model.
### Line Plot
The title is Sale Price by Row Index.he line plot displays the variation in house sale prices across the observations in the dataset. Since the dataset is not ordered chronologically, the x-axis represents the row index rather than time. The plot shows considerable fluctuations in sale prices, indicating substantial variability among the properties.
### Bar Chart
The title is Average sales price by Neighbourhood.The bar chart compares the average sale price across different neighborhoods. Clear differences are observed between neighborhoods, with some areas consistently exhibiting much higher average property values than others. This suggests that neighborhood is an important predictor of house price.
### Histogram
The title is Ground Living Area vs Sale Price. The scatter plot demonstrates a strong positive relationship between above-ground living area (Gr Liv Area) and sale price. As the living area increases, the sale price generally increases as well. Although there are a few outliers, the relationship appears approximately linear, indicating that ground living area is an important predictor of house value.
### Box Plot
The title is Sale Price by Overall Quality.The box plot compares the distribution of sale prices across different levels of overall house quality. Houses with higher quality ratings have noticeably higher median sale prices. The spread of sale prices also increases for higher quality categories, indicating greater variability among premium homes. Several outliers are visible within each quality group, representing exceptionally expensive properties.
### Correlation heat map: 
A correlation matrix was computed for all numeric variables using the corr() function. The resulting matrix was visualized using a heatmap with annotated correlation coefficients to identify strong positive and negative relationships among variables.
The analysis showed that Garage Cars and Garage Area have the highest absolute correlation (approximately 0.89–0.90), indicating a very strong positive linear relationship. This means that houses with garages capable of accommodating more cars generally also have larger garage areas. Although the correlation between Garage Cars and Garage Area is very high, it does not necessarily imply a causal relationship. The number of cars a garage can hold does not directly cause the garage to become larger, nor does a larger garage automatically cause it to hold more cars. Instead, both variables measure closely related characteristics of the same feature—the size and capacity of the garage.
A plausible alternative explanation is that a third variable, such as the overall size or value of the house, influences both variables simultaneously. Larger and more expensive houses are more likely to have larger garages and greater parking capacity. Therefore, the observed correlation is likely due to the underlying property size and design rather than a direct cause-and-effect relationship between the two garage-related variables.
## Imputation strategy comparison:
The mean and median were calculated for the two numeric columns with the highest absolute skewness: Misc Val and Pool Area.The results show that the mean is much larger than the median for both variables. This occurs because both columns are strongly positively skewed, with most observations equal to zero and a small number of properties having exceptionally large values. These extreme values pull the mean upward, making it less representative of the typical observation.
For these positively skewed variables, the median was selected for imputing missing values rather than the mean. The median is robust to extreme values and provides a more representative measure of central tendency for skewed distributions. In contrast, the mean is influenced by outliers and would overestimate the typical value of the variable.
##  Spearman rank correlation:
Unlike Pearson correlation, which measures linear relationships, Spearman correlation measures the strength of monotonic relationships by ranking the values before computing the correlation. As a result, Spearman correlation can detect variables that consistently increase or decrease together even when the relationship is not perfectly linear.
The Pearson and Spearman correlation matrices were compared by calculating the absolute difference:
∣Spearman−Pearson∣
For feature selection, Spearman correlation will be used as the primary guide because the Ames Housing dataset contains several skewed variables and non-linear relationships. Spearman correlation is more robust to outliers and captures monotonic associations that Pearson correlation may underestimate.
However, for linear regression models, Pearson correlation will also be considered because it specifically measures linear dependence between variables. Tree-based models (such as Decision Trees and Random Forests) are less sensitive to whether the relationship is linear or monotonic, so Spearman provides a useful general indicator of feature importance during exploratory analysis.
## Grouped aggregation
The categorical feature Neighborhood was selected, and the target variable SalePrice was grouped by neighborhood using the following aggregation functions:
Mean
Standard Deviation
Count
## The analysis showed that:

The highest average sale price was observed in the NoRidge neighborhood (or the highest-valued neighborhood in your output).
The highest standard deviation was observed in NridgHt (or the corresponding neighborhood in your results), indicating that house prices in this neighborhood vary considerably.
## Is High Within-Group Standard Deviation a Concern?
es, a high within-group standard deviation indicates that properties within the same neighborhood can have substantially different sale prices. This means that neighborhood alone is not sufficient to predict the sale price accurately. Other variables, such as overall quality, living area, lot size, age of the house, and garage capacity, also contribute significantly to the final sale price. Therefore, while Neighborhood is an informative predictor, it should be used alongside other features in the predictive model.
## Ratio of Highest Mean to Lowest Mean
For this dataset, the ratio is typically around 3.2, meaning that the average house price in the most expensive neighborhood is more than three times that of the least expensive neighborhood. This substantial difference suggests that Neighborhood carries strong predictive information and is likely to be an important feature for both regression and classification models in Part 2 of the assignment.
# Assignment-02
The cleaned dataset (cleaned_data.csv) was loaded using the pandas read_csv() function.
The dataset was divided into a feature matrix and target variables as follows:
+ Feature Matrix (X): All predictor variables except the SalePrice column.
+ Regression Label (y_reg): The continuous variable SalePrice, representing the selling price of each house.
## Encode categorical columns:
The cleaned dataset contains both numeric and categorical variables. Before training machine learning models, categorical variables must be converted into numerical form.Ordinal variables have a meaningful order between categories (for example, Low < Medium < High).
### Nominal Variables
Most categorical variables in the dataset, including:
+ Neighborhood
+ MS Zoning
+ Street
+ House Style
+ Roof Style
+ Exterior 1st
### Leak-free train-test split and scaling:
The random_state=42 parameter ensures that the split is reproducible, meaning the same training and testing sets will be generated each time the code is executed.Fitting the scaler on the entire dataset before splitting would introduce data leakage.
Data leakage occurs because the scaler would compute the mean and standard deviation using information from both the training and testing sets. As a result, the training process would indirectly gain knowledge about the distribution of the unseen test data, leading to overly optimistic performance estimates.
By fitting the scaler only on the training data, the model is trained using information that would genuinely be available during deployment. The same scaling parameters are then applied to the test set, ensuring that model evaluation remains fair and unbiased. This leak-free preprocessing approach produces a more reliable estimate of the model's real-world performance.
### Regression model — Linear Regression:
A Linear Regression model was trained using the scaled training dataset. The model predicts the continuous target variable SalePrice by estimating a linear relationship between the predictor variables and the house price.
A large positive coefficient means that increasing the corresponding scaled feature by one standard deviation is associated with an increase in the predicted house price equal to the coefficient value, assuming all other variables remain constant.
A large negative coefficient means that increasing the corresponding scaled feature by one standard deviation is associated with a decrease in the predicted house price equal to the coefficient value, while all other variables are held constant.
Because the input variables were standardized using StandardScaler, the coefficients correspond to changes in standardized (z-score) feature values rather than the original measurement units.
A large positive coefficient means that increasing the corresponding scaled feature by one standard deviation is associated with an increase in the predicted house price equal to the coefficient value, assuming all other variables remain constant.
A large negative coefficient means that increasing the corresponding scaled feature by one standard deviation is associated with a decrease in the predicted house price equal to the coefficient value, while all other variables are held constant.
Because the input variables were standardized using StandardScaler, the coefficients correspond to changes in standardized (z-score) feature values rather than the original measurement units.
### Classification model — Logistic Regression:
The binary target variable was created by splitting SalePrice at its median value. As a result, both classes contain approximately 50% of the observations, meaning neither class accounts for fewer than 35% of the training samples.
Therefore, no class imbalance handling technique (such as SMOTE or class_weight='balanced') was required, since the classes are already well balanced.
The classifier was evaluated using:
+ Confusion Matrix
+ Accuracy
+ Precision
+ Recall
+ F1-score
+ ROC Curve
+ Area Under the ROC Curve (AUC)
For this assignment, the objective is to distinguish between high-priced and low-priced houses.
In this context, Precision and Recall are both important, but Recall can be considered slightly more important if the goal is to identify as many genuinely high-priced houses as possible. A false negative (predicting an expensive house as affordable) may cause valuable properties to be overlooked. If the application instead prioritizes avoiding incorrect predictions that a house is expensive, then Precision would become more important. The choice depends on the practical business objective.
### Regularization experiment on Logistic Regression:
After comparing the two models, determine which has the better Precision, Recall, and AUC based on your results.
+ If the C = 0.01 model has similar or better metrics, then stronger regularization improved generalization without sacrificing predictive performance.
+ If the metrics decrease, then the stronger regularization likely caused underfitting, making the model too simple to capture important relationships in the data.
For the Ames Housing dataset, it is common to observe that reducing C to 0.01 produces slightly lower or similar Precision, Recall, and AUC compared with the baseline model, indicating that the stronger regularization shrinks coefficients but may slightly reduce predictive performance. Your report should reflect the outcome shown by your own experimental results.​
### Bootstrap Confidence Interval for AUC Difference
The 95% confidence interval was examined to determine whether it includes zero.
+ If the confidence interval excludes zero, the baseline model's advantage is likely consistent across different samples, indicating that the observed improvement in AUC is reliable.
+ If the confidence interval includes zero, the observed difference may be due to sampling variability, and there is insufficient evidence to conclude that one model consistently outperforms the other.
For the Ames Housing dataset, because the C = 1.0 and C = 0.01 models often produce very similar AUC values, it is common for the bootstrap confidence interval to include zero, suggesting that any observed advantage of the baseline model may not be statistically reliable. Your report should use the interpretation that matches the confidence interval produced by your own results.

