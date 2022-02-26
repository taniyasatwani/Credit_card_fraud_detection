# Credit Card Fraud Detection 
A machine learning/ deep learning model that classifies transactions as fraud or not. 

## Inspiration


Credit card fraud is any dishonest act and behaviour to obtain information without the proper authorization from the account holder for financial gain. For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

In the banking industry, credit card fraud detection using machine learning is not just a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees, and denials of legitimate transactions.

## Dataset

The dataset is taken from the [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

## Data Dictionary

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.


## Project Pipeline

**Get Started**
1. Used machine learning models(logistic regresion, decision tree, random forest, xgboost) to classify transactions. Run `credit_card_fraud_detection.ipynb`.

   A. On the original unbalance data.
   
   B. Did sampling(undersampling, oversampling, SMOTE, AdaSyn) on unbalance data.
   
2. Used deep learning model(autoencoder) to classify transaction. Run `credit_card_fraud_detection_using_autoencoders.ipynb`.

The project pipeline can be briefly summarized in the following five steps:

Solution approach
1. Data understanding and exploring
2. Data cleaning
• Handling missing values 
• Outliers treatment
3. Exploratory data analysis
• Univariate analysis
• Bivariate analysis
4. Prepare the data for modelling 
• Check the skewness of the data and mitigate it for fair analysis
• Handling data imbalance as we see only 0.172% records are the fraud 
transactions
5. Split the data into train and test set
• Scale the data (normalization)
6. Model building
• Train the model with various algorithm such as Logistic regression, SVM, 
Decision Tree, Random forest, XGBoost etc.
• Tune the hyperparameters with Grid Search Cross Validation and find the 
optimal values of the hyperparameters
7. Model evaluation 
• As we see that the data is heavily imbalanced, Accuracy may not be the 
correct measure for this particular case
• We have to look for a balance between Precision and Recall over Accuracy
• We also have to find out the good ROC score with high TPR and low FPR in 
order to get the lower number of misclassifications
