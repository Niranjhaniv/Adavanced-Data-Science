<h1>Part 1 - Model Design and Building</h1>

This part of the assignment designs and build prediction model for a bank campaign performed to predict whether a customer will subscribe to the term deposit provided by the Bank or not.

<h2> DataCleaning & EDA.pynb </h2>

(a.) Data has been pulled from Amazon S3 to the local machine

(b.) Dataset is read from the downloaded file and then ingested into Pandas

(c.)There are 21 features in the dataset that contains 11 categorical and 10 numerical variables. Dataset is analysed for any outliers and extreme values to avoid uncessary incline towards it during prediction. After Data cleaning , the dataset shape reduces from **(41188, 21)** to **(38582, 20)**

(d.) The cleaned dataset is then exported to csv 

(e.) Exploratory data analysis is performed on the dataset to read the data pattern and analysis conclusion has been updated at the end of the notebook

<h2>FeatureEngineering.ipynb</h2>

(a.) All the categorical features are on hot encoded so that it can be used for mathematical analysis

(b.) Extra columns are added for analysis purpose

(c.) The output dataset is then exported to csv

<h2>FeatureSelection_ADS.ipynb</h2>

(a.) Since the features after feature engineering increased from 21 to 65. Feature selection techniques are perform.

(b.) Features a re selected on the basis of output from Recrusive feature elimination and Feature importance techniques.

<h2>MachineLearning.ipynb</h2>

(a.) With the csv from feature engineering notebook, the dataset is cross validated with 13 different classifiers.

(b.) Best three classifier with highest cross validation scores are taken

(c.) All the three classifier are seperaetly trained with train dataset and predicted with test dataset

<h2>ModelSelection.ipynb</h2>

(a.) With selected features hyperparameter tunning is done using Grid search for three different model:

                    1. MLP Classifier
                    
                    2. SGD Classifier
                    
                    3. Logistic Regression 

<h2>Pickle LogisticRegression.ipynb</h2>

(a.) Model are redesigned with hypertuned parameters along with selected features

(b.) Entier model is pickled

(c.) Error metrics are calculated and exported as csv for LogisticRegression

<h2>Pickle MLPClassifier.ipynb</h2>

(a.) Model are redesigned with hypertuned parameters along with selected features

(b.) Entier model is pickled

(c.) Error metrics are calculated and exported as csv for MLPClassifier

<h2>Pickle SGDClassifier.ipynb</h2>

(a.) Model are redesigned with hypertuned parameters along with selected features

(b.) Entier model is pickled

(c.) Error metrics are calculated and exported as csv for SGDClassifier

<h2>Upload.ipynb</h2>

(a.) A bucket is created in S3 using boto3

(b.) All three pickle file along with repective error metric csv are uploaded to amazon s3 bucket

<h2/>Pipeline</h2>

The entire model has been pipelined using **Luigi**

<h2>Docker</h2>

The final pipeline is docerized

The commands to build and run the docker image are

(a.) docker build -f Dockerfile .

(b.) docker run -e ak=<AWS_ACCESS_KEY> -e sk=<AWS_SECRET_KEY> -t -i "username"/"reponame"


**The models in Part 1 will be deployed as a web application in Part 2 with user interface**
