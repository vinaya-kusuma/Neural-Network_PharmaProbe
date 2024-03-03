# Neural-Network_PharmaProbe

## Overview

In today's digital age, patients have unprecedented access to information about medications through online platforms where they can share their experiences and reviews of different drugs. This vast amount of data presents both opportunities and challenges for healthcare professionals, policymakers, and pharmaceutical companies. By analyzing drug review datasets, we can extract valuable insights to improve patient care, identify emerging trends, and support evidence-based decision-making.

Our project aims to leverage advanced data analytics techniques to extract meaningful insights from Drug Review datasets. By analyzing patient reviews, ratings, and associated metadata, we seek to uncover trends, patterns, and correlations that can shed light on medication efficacy, safety, and patient satisfaction. 

## Analyses Conducted

### 1. Correlation Analysis

- **Objective:** Investigate the correlation between drug ratings and useful count values.
- **Method:** Conducted correlation analysis to examine the relationship between drug ratings and useful count values.
- **Result:** Found a clear positive correlation between drug ratings and useful count values, indicating that drugs with higher ratings tend to have higher useful counts.

### 2. Rating Distribution Analysis

- **Objective:** Analyze the distribution of drug ratings and identify top-rated drugs.
- **Method:** Plotted a histogram of ratings to visualize the frequency of each rating value across all drugs. Identified top 20 rated drugs with a perfect rating of 10/10.
- **Result:** Visualized the distribution of ratings and highlighted the top-rated drugs in the dataset.

### 3. Seasonal Impact Analysis

- **Objective:** Determine if seasonal variations impact drug rating values.
- **Method:** Calculated the average ratings received per season to explore potential trends or patterns in ratings across different seasons.
- **Result:** Investigated whether there are noticeable fluctuations in ratings based on the season in which the drug reviews were made.

# Project Outline

<img width="657" alt="image" src="https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/81578500/b763e3b1-c4ca-4f85-92d7-5f99ce9ce16e">


# Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is an essential approach in data analysis that involves identifying general patterns, outliers, and unexpected features in the data. It serves as a crucial first step in understanding and processing the dataset.

## Overview

After loading the data, we conducted a series of steps to clean and explore the dataset:

## Data Cleaning

We began by checking for NaN values in the conditions and other columns and removed records containing NaNs. Subsequently, we standardized the language in the review text by converting it to numerical representations. Additionally, we removed punctuations and special characters and converted all review text to lowercase to facilitate further processing.

## Outlier Detection and Removal

To identify outliers, we created a new column to count the number of words in the reviews (review_length) and removed outliers in review length, useful_count, and ratings. This step ensured that our analysis focused on reliable data points.

## Analysis of Relationships

We examined the relationships between review length, useful count, and ratings by plotting graphs to visualize these relationships. Furthermore, we explored the seasonality of user ratings to gain insights into potential patterns or trends over time.

## Analysis of Categorical Columns

We analyzed categorical columns in the dataset, particularly focusing on sickness conditions. Initially, we identified a total of 691 conditions but narrowed down our analysis to the top 10 conditions for further exploration.

## Visualization

Leveraged Tableau to create interactive visualizations illustrating the correlation, rating distribution, top-rated drugs, and seasonal trends, enhancing data exploration and interpretation. We also utilized various visualization techniques to present our findings effectively. For instance, we plotted graphs to illustrate the relationships between variables, such as the correlation between useful count and review length. Additionally, we used a pie chart to visualize the occurrence of the top 10 sickness conditions in the dataset, highlighting the most prevalent conditions being treated by drugs.

## Conclusion

This comprehensive EDA approach provided valuable insights into the dataset, enabling us to identify patterns, outliers, and relationships between variables. The findings serve as a solid foundation for further data exploration, modeling, and decision-making processes.

## Useful files of models
* Neural network condition cluster model
  * [Neural_network_condtion_cluster_model/BERT_Tokenization.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_condtion_cluster_model/BERT_Tokenization.ipynb)
  * [Neural_network_condtion_cluster_model/Clean_Neural_Net_No_PCA_for_Testing.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_condtion_cluster_model/Clean_Neural_Net_No_PCA_for_Testing.ipynb)
  * [Neural_network_condtion_cluster_model/With_Sentiment_.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_condtion_cluster_model/With_Sentiment_.ipynb)
* Neural network sentiment analysis model
  * [Neural_network_Sentiment_Analsys_model/OpenAI_Review_Sentiment.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_Sentiment_Analsys_model/OpenAI_Review_Sentiment.ipynb)
  * [Neural_network_Sentiment_Analsys_model/PCA.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_Sentiment_Analsys_model/PCA.ipynb)
  * [Neural_network_Sentiment_Analsys_model/Sentiment_Analysis.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_Sentiment_Analsys_model/Sentiment_Analysis.ipynb)
  * [Neural_network_Sentiment_Analsys_model/Sentiment_Analysis_PCA.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_Sentiment_Analsys_model/Sentiment_Analysis_PCA.ipynb)
* Neural network drug name cluster model
  * [Neural_network_Drug_name_cluster_model/Neural_Network_with_DrugNameTarget.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/blob/main/Neural_network_Drug_name_cluster_model/Neural_Network_with_DrugNameTarget.ipynb)
    
## Tableau links
* [Drug_name_dashboard](https://public.tableau.com/app/profile/pallavi.tripathi/viz/Neural_NEtwork_Drug_name/Drug_name_dashboard)
* [Rating_dashboard](https://public.tableau.com/app/profile/ranjini.rao1648/viz/Neural-Network_PharmaProbe_Rating/correlationbetweentheusefulCountandratingofdrugs?publish=yes)
* [Scatter plot with linear regression](https://public.tableau.com/app/profile/madhavi.pandey/viz/Project4_17091757165050/ScatterPlotWithLinearRegression)
* [Histrogram of usefulcounts](https://public.tableau.com/app/profile/madhavi.pandey/viz/Project_4_17091757836620/HistogramofusefulCount)

## Neural Network Drug Name cluster model
**Problem**: To check if we can predict the drug cluster for a drug with high confidence using features such as 
1) embedded review,
2) ratings,
3) useful_count and
4) drug name dummy variable.

### File location
[Neural_network_Drug_name_cluster_model/Neural_Network_with_DrugNameTarget.ipynb](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/tree/main/Neural_network_Drug_name_cluster_model)

### Model training approaches
We used word2vec google library to embed the drug name.This library organizes words based on their meanings, grouping similar words together closely in a geometric space. Then we used K-Means to group these words into clusters.
By using all available features (including drug name) and a similar Neural Net structure, our team was able to classify the Drug Name cluster with 95.41% accuracy.

Such a high prediction accuracy asserts that clustering was done appropriately and if a prediction is made for a new Drug, it is highly likely for it to be placed in correct cluster.

<img width="942" alt="Screenshot 2024-02-29 at 4 00 15 PM" src="https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/139268721/d429734d-f6c5-4606-a3aa-7e67a4006527">

We used different approaches for model training as shown above.
1. We used *embedded review, condition-dummy-variable and useful count* to predict the drug cluster. This approach gave accuracy of 41.7 %
2. We added *hyper-parameters* in the next try but that didn’t help a lot in accuracy. Accuracy was 40.7%
3. We then used another csv file and introduced *sentiment analysis* as feature. There was again not a lot of impact on accuracy
4. Lastly, we introduced *drug-name dummy variable* as a feature and this helped to propel the accuracy to 95.4%.

### Test data classification
<img width="830" alt="Screenshot 2024-02-29 at 4 01 19 PM" src="https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/139268721/4e8c2ddd-941f-4afa-81ee-defdbd270230">

As we can see here that accuracy is pretty high (greater than 88%) for all the cluster. This shows that the clustering we did for drugs was appropriate.

### Usefulness of the model

Given we can now predict drug cluster with high accuracy, we will have high confidence to put a new drug into pre-existing clusters. 
1. This approach can particularly useful for pharma companies who would want to group similar drugs together for storage and/or disposal purposes.
2. This can also be useful for seller companies to advertise/promote/recommend similar drugs if a customer is interested in particular type of drug.


# Neural Network for Classifying Conditions

This neural network project aims to classify the condition cluster based on drug reviews. The project utilizes various techniques such as tokenization, embedding, data preprocessing, neural network classification, hyperparameter tuning, and feature engineering.

Steps:
Step 1: Tokenization and Embedding with BERT
Utilized BERT to tokenize and embed the review column using Pandas DataFrames.
BERT (Bidirectional Encoder Representations from Transformers) is employed to convert the textual data into numerical vectors, capturing the semantic meaning of words in the reviews.

Step 2: Neural Network Classification
Split the data into test and train sets using scikit-learn.
Scaled the data to ensure consistent feature ranges.
Implemented a neural network using TensorFlow to classify the condition cluster based on the embedded reviews.

Step 3: Assessment of Classification Accuracy
Used NumPy to assess the accuracy of the classifications made by the neural network.
Step 4: Hyperparameter Tuning with Keras Tuner
Employed Keras Tuner to tune the hyperparameters of the neural network.
Utilized the same features and target as in Step 2 to optimize the model's performance.

Step 5: Feature Engineering with Additional Features
Expanded the feature set by including additional features such as review length, drug cluster, rating, and useful count.
Repeated Steps 2-4 with the augmented feature set to enhance the model's predictive capability.

Step 6: Feature Engineering with Drug Name
Added drug name as a feature and removed drug cluster.
Utilized Pandas' get_dummies function to convert drug names into dummy columns for inclusion in the feature set.
Repeated Steps 2-4 with the updated feature set, including drug names.

Step 7: Sentiment Analysis as an Additional Feature
Incorporated sentiment analysis of the reviews as an additional feature.
Created dummy columns for sentiment analysis results.
Repeated Steps 2-4 with the augmented feature set, including sentiment analysis results.

# Neural Network for Sentimental Analysis
This model aims to classify the sentiment of the drug reviews as Positive and negative. 

![image](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/81578500/c213095e-2db6-4606-a44c-535a0d3e3b2d)

Step 1: Data Preparation 
OpenAI chat completion API was prompted to label each of the reviews with sentiments as positive, negative and neutral.
The labeling was done in bacthes with batch size of 500.
The labeled reviews are written into CSV file

Step 2: Tokenization and Embedding with BERT
Utilized BERT to tokenize and embed the review column using Pandas DataFrames.
BERT (Bidirectional Encoder Representations from Transformers) is employed to convert the textual data into numerical vectors, capturing the semantic meaning of words in the reviews.

Step 3: Neural network Model
To balance the data, drug reviews with positive and negative sentiments were considered, while those with neutral label were dropped.

With input features as review embeddings and target as sentiment, a neural network model was built and trained.The accuracy of the model was 64.36%

<img width="561" alt="image" src="https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/81578500/4559d423-8fd6-4a35-9f7f-0fa930c477e6">

With 2 more features - ratings and useful counts incorporated,the accuracy of the model was at 79.73% and with hyperparameter tuning, model accuracy was at 79.91%

dimensionality reduction on review embeddings using PCA

-For 95% variance, the dimensionality reduction of review embeddings was from 768 to 330 components as seen from the cumulative variance plot.
![image](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/81578500/62c31cc2-5c9a-4c02-ab65-b4062243180d)

Accuracy of the redesigned model with reduced dimensionality was 80.62%

Step 4: Performance evaluation - stratified k-fold cross-validation with k set to 10

From this metrics, we see that the model has an average accuracy of 80.1% with standard deviation of .0052

![image](https://github.com/ranjini-rao/Neural-Network_PharmaProbe/assets/81578500/1768c5d8-2b25-408c-96af-859ea8564ede)


# Conclusion:

This project demonstrates the use of neural networks for classifying condition clusters based on drug reviews. By leveraging advanced techniques such as tokenization, embedding, hyperparameter tuning, and feature engineering, the model's predictive accuracy and robustness are enhanced. Future improvements may involve exploring alternative neural network architectures, experimenting with different embeddings, and further refining feature engineering strategies.






