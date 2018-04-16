<h1>Assignment -3</h1>

<h2> Problem Statement </h2>

AdaptiveAlgo Systems Inc. has invited you to be partners in helping them implement datascience solutions. Your company was selected after rigorous review and is held in high esteem for the quality data science solutions you have been developing. However, the AdaptiveAlgo is concerned if your team of three can deliver high performance solutions in a timely manner. Note that AdaptiveAlgo wants to make a decision on which team they want to partner with based on the deliverable you submit by April 13th 11.59pm. The Dataset will be made available through Amazon S3. Since AdaptiveAlgo has all solutions on the cloud, you should also implement all solutions on the cloud. You have a choice of cloud.

<h3> Part - 1 Part1: Model design and building</h3>

1. The dataset is on Amazon S3. Access the data assigned to you from S3
2. You should build a pipeline using Luigi/Airflow/Sklearn (See the google link for your team’s allocated method. This pipeline incorporates:
  a. Data ingestion into Pandas
  b. Cleanup the Data if needed
  c. Exploratory Data Analysis with Plotly/seaborn/matplotlib
  d. Feature Engineering on the data
  e. Feature Selection or any transformation on the dataset
  f. Run Different Machine learning models (at-least 3) for the problem assigned
  g. Get the Accuracy and Error metrics for all the models and store them in a csv file with Ranking of the models
  h. Pickle all the models
  i. Upload the error metric csv and models to s3 bucket
3. Dockerize this pipeline using Repo2Docker or write your own docker file
4. Note:
(a.) Properly document your code
(b.) Use Python classes and functions when needed for replicability and reuse.
(c.) You should try and use configuration files when possible to ensure you can make modifications and your solution is generic.
(d.) You should also write a comprehensive Readme.md to detail your design,implementation, results and analysis
(e.) Use any other Python package when needed


<h3> Part2: Model Deployment </h3>

1. Create a Web application using Flask that uses the models created (in Pickle format) in Part1 and stored on S3
2. Build a web page which takes user inputs. The application should allow submission of data for prediction via Forms as well as REST Api calls using JSON
3. The application should allow submission on single record of data as well as batch of records data upload to get single/bulk responses.
4. The Result should be provided to the user as a csv file and a table with results should be displayed
5. You need to use the models saved in S3 to run your models.
6. Create Unit tests for the user and test your application.
7. Dockerize this using repo2docker or write your own docker file. Whenever your run your docker image, your application should get the latest models from S3 and do predictions on all the three (or any number of models you developed) and present outputs.
8. Note that your webapp should get the latest models whenever the models change. You implement this using Amazon Lambda. 
9. When you have more than 10 inputs, use Dask to setup a cluster and divide the load of computation
10. Write a Readme.md detailing your application
