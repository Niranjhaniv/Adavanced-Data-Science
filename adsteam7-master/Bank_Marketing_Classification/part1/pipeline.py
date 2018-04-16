import luigi
import csv
import zipfile,io,os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import argmax
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from urllib.request import urlopen
import os
import boto3
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from io import BytesIO
import pickle
import datetime
import time
from zipfile import ZipFile
from sklearn.linear_model import SGDClassifier

class DataCleaning(luigi.Task):

    def output(self):
        return luigi.LocalTarget("LuigiDC.csv")

    def run(self):
        url = urlopen("https://s3.amazonaws.com/assignment3datasets/bank.zip")
        path = str(os.getcwd())
        z = ZipFile(BytesIO(url.read()))
        with z as zfile:
            zfile.extractall(path)
        df = pd.read_csv("bank-additional/bank-additional-full.csv", sep=";")
        df_test = pd.read_csv("bank-additional/bank-additional.csv", sep=";")
        df=df.rename(columns={'y':'result'})
        df= df.rename(columns={'emp.var.rate':'emp_var_rate','cons.price.idx':'cons_price_idx','cons.conf.idx':'cons_conf_idx','nr.employed':'nr_employed'})
        df_test=df_test.rename(columns={'y':'result'})
        df_test= df_test.rename(columns={'emp.var.rate':'emp_var_rate','cons.price.idx':'cons_price_idx','cons.conf.idx':'cons_conf_idx','nr.employed':'nr_employed'})
        df.loc[df.pdays == 999, 'pdays'] = 0
        df_test.loc[df_test.pdays == 999, 'pdays'] = 0
        df.drop("duration", axis=1, inplace=True)
        df_test.drop("duration", axis=1, inplace=True)
        cam = df.campaign
        data_1 = [x for x in cam if (x > np.mean(cam) - 3 * np.std(cam))]
        cam = [x for x in data_1 if (x < np.mean(cam) + 3 * np.std(cam))]
        cons = df.cons_conf_idx
        data_1 = [x for x in cons if (x > np.mean(cons) - 3 * np.std(cons))]
        cons = [x for x in data_1 if (x < np.mean(cons) + 3 * np.std(cons))]
        df = df[df.campaign.isin(cam)][df.cons_conf_idx.isin(cons)]
        cam = df_test.campaign
        data_1 = [x for x in cam if (x > np.mean(cam) - 3 * np.std(cam))]
        cam = [x for x in data_1 if (x < np.mean(cam) + 3 * np.std(cam))]
        cons = df_test.cons_conf_idx
        data_1 = [x for x in cons if (x > np.mean(cons) - 3 * np.std(cons))]
        cons = [x for x in data_1 if (x < np.mean(cons) + 3 * np.std(cons))]
        df_test = df_test[df_test.campaign.isin(cam)][df_test.cons_conf_idx.isin(cons)]
        df_test.to_csv("LuigiDCTest.csv",index=False)
        df.to_csv(self.output().path,index=False)

class FeatureEngineering(luigi.Task):
    
    def requires(self):
        yield DataCleaning() 
    
    def output(self):
        return luigi.LocalTarget("LuigiFE.csv")
    
    def run(self):
        df=pd.read_csv(DataCleaning().output().path)
        df_test = pd.read_csv("LuigiDCTest.csv")
        # df_test =df_test.drop('Unnamed: 0',axis=1)
        # df=df.drop('Unnamed: 0',axis=1)
        df.result.replace(('yes', 'no'), (1, 0), inplace=True)
        df_test.result.replace(('yes', 'no'), (1, 0), inplace=True)
        label_encoder = LabelEncoder()
        job = label_encoder.fit_transform(df['job'])
        marital = label_encoder.fit_transform(df['marital'])
        edu = label_encoder.fit_transform(df['education'])
        defa = label_encoder.fit_transform(df['default'])
        hous = label_encoder.fit_transform(df['housing'])
        loan = label_encoder.fit_transform(df['loan'])
        contact = label_encoder.fit_transform(df['contact'])
        month = label_encoder.fit_transform(df['month'])
        day = label_encoder.fit_transform(df['day_of_week'])
        pout = label_encoder.fit_transform(df['poutcome'])
        onehot_encoder = OneHotEncoder(sparse=False)
        job = job.reshape(len(job), 1)
        marital = marital.reshape(len(marital), 1)
        edu = edu.reshape(len(edu), 1)
        defa = defa.reshape(len(defa), 1)
        hous = hous.reshape(len(hous), 1)
        loan = loan.reshape(len(loan), 1)
        contact = contact.reshape(len(contact), 1)
        month = month.reshape(len(month), 1)
        day = day.reshape(len(day), 1)
        pout = pout.reshape(len(pout), 1)
        newjob = onehot_encoder.fit_transform(job)
        newmarital = onehot_encoder.fit_transform(marital)
        newedu = onehot_encoder.fit_transform(edu)
        newdefa = onehot_encoder.fit_transform(defa)
        newhous = onehot_encoder.fit_transform(hous)
        newloan = onehot_encoder.fit_transform(loan)
        newcontact = onehot_encoder.fit_transform(contact)
        newmonth = onehot_encoder.fit_transform(month)
        newDay = onehot_encoder.fit_transform(day)
        newpout = onehot_encoder.fit_transform(pout)

        df.drop(['job','marital','education','default','housing','loan', 'month','contact','poutcome','day_of_week'], axis=1, inplace=True)
        df['housemaid'] = pd.Series(newjob[:,0], index=df.index)
        df['services'] = pd.Series(newjob[:,1], index=df.index)
        df['admin.'] = pd.Series(newjob[:,2], index=df.index)
        df['blue-collar'] = pd.Series(newjob[:,3], index=df.index)
        df['technician'] = pd.Series(newjob[:,4], index=df.index)
        df['retired'] = pd.Series(newjob[:,5], index=df.index)
        df['management'] = pd.Series(newjob[:,6], index=df.index)
        df['unemployed'] = pd.Series(newjob[:,7], index=df.index)
        df['self-employed'] = pd.Series(newjob[:,8], index=df.index)
        df['job_unknown'] = pd.Series(newjob[:,9], index=df.index)
        df['entrepreneur'] = pd.Series(newjob[:,10], index=df.index)
        df['student'] = pd.Series(newjob[:,11], index=df.index)

        df['married'] = pd.Series(newmarital[:,0], index=df.index)
        df['single'] = pd.Series(newmarital[:,1], index=df.index)
        df['divorced'] = pd.Series(newmarital[:,2], index=df.index)
        df['marital_unknown'] = pd.Series(newmarital[:,3], index=df.index)

        df['basic_4y'] = pd.Series(newedu[:,0], index=df.index)
        df['high.school'] = pd.Series(newedu[:,1], index=df.index)
        df['basic_6y'] = pd.Series(newedu[:,2], index=df.index)
        df['basic_9y'] = pd.Series(newedu[:,3], index=df.index)
        df['professional_course'] = pd.Series(newedu[:,4], index=df.index)
        df['education_unknown'] = pd.Series(newedu[:,5], index=df.index)
        df['university_degree'] = pd.Series(newedu[:,6], index=df.index)
        df['illiterate'] = pd.Series(newedu[:,7], index=df.index)

        df['default_no'] = pd.Series(newdefa[:,0], index=df.index)
        df['default_unknown'] = pd.Series(newdefa[:,1], index=df.index)
        df['default_yes'] = pd.Series(newdefa[:,2], index=df.index)

        df['housing_no'] = pd.Series(newhous[:,0], index=df.index)
        df['housing_yes'] = pd.Series(newhous[:,1], index=df.index)
        df['housing_unknown'] = pd.Series(newhous[:,2], index=df.index)

        df['loan_no'] = pd.Series(newloan[:,0], index=df.index)
        df['loan_yes'] = pd.Series(newloan[:,1], index=df.index)
        df['loan_unknown'] = pd.Series(newloan[:,2], index=df.index)

        df['telephone'] = pd.Series(newcontact[:,0], index=df.index)
        df['cellular'] = pd.Series(newcontact[:,1], index=df.index)


        df['mar'] = pd.Series(newmonth[:,0], index=df.index)
        df['apr'] = pd.Series(newmonth[:,1], index=df.index)
        df['may'] = pd.Series(newmonth[:,2], index=df.index)
        df['jun'] = pd.Series(newmonth[:,3], index=df.index)
        df['jul'] = pd.Series(newmonth[:,4], index=df.index)
        df['aug'] = pd.Series(newmonth[:,5], index=df.index)
        df['sep'] = pd.Series(newmonth[:,6], index=df.index)
        df['oct'] = pd.Series(newmonth[:,7], index=df.index)
        df['nov'] = pd.Series(newmonth[:,8], index=df.index)
        df['dec'] = pd.Series(newmonth[:,9], index=df.index)

        df['mon'] = pd.Series(newDay[:,0], index=df.index)
        df['tue'] = pd.Series(newDay[:,1], index=df.index)
        df['wed'] = pd.Series(newDay[:,2], index=df.index)
        df['thu'] = pd.Series(newDay[:,3], index=df.index)
        df['fri'] = pd.Series(newDay[:,4], index=df.index)

        df['nonexistent'] = pd.Series(newpout[:,0], index=df.index)
        df['failure'] = pd.Series(newpout[:,1], index=df.index)
        df['success'] = pd.Series(newpout[:,2], index=df.index)
        
        df['jan'] = np.nan
        df['feb'] = np.nan
        df=df.fillna(0)
        
        arrange = df.columns.tolist()
        arrange.remove('result')
        arrange.append('result')
        df = df[arrange]

        label_encoder = LabelEncoder()
        job = label_encoder.fit_transform(df_test['job'])
        marital = label_encoder.fit_transform(df_test['marital'])
        edu = label_encoder.fit_transform(df_test['education'])
        defa = label_encoder.fit_transform(df_test['default'])
        hous = label_encoder.fit_transform(df_test['housing'])
        loan = label_encoder.fit_transform(df_test['loan'])
        contact = label_encoder.fit_transform(df_test['contact'])
        month = label_encoder.fit_transform(df_test['month'])
        day = label_encoder.fit_transform(df_test['day_of_week'])
        pout = label_encoder.fit_transform(df_test['poutcome'])
        onehot_encoder = OneHotEncoder(sparse=False)
        job = job.reshape(len(job), 1)
        marital = marital.reshape(len(marital), 1)
        edu = edu.reshape(len(edu), 1)
        defa = defa.reshape(len(defa), 1)
        hous = hous.reshape(len(hous), 1)
        loan = loan.reshape(len(loan), 1)
        contact = contact.reshape(len(contact), 1)
        month = month.reshape(len(month), 1)
        day = day.reshape(len(day), 1)
        pout = pout.reshape(len(pout), 1)
        newjob = onehot_encoder.fit_transform(job)
        newmarital = onehot_encoder.fit_transform(marital)
        newedu = onehot_encoder.fit_transform(edu)
        newdefa = onehot_encoder.fit_transform(defa)
        newhous = onehot_encoder.fit_transform(hous)
        newloan = onehot_encoder.fit_transform(loan)
        newcontact = onehot_encoder.fit_transform(contact)
        newmonth = onehot_encoder.fit_transform(month)
        newDay = onehot_encoder.fit_transform(day)
        newpout = onehot_encoder.fit_transform(pout)

        df_test.drop(['job','marital','education','default','housing','loan', 'month','contact','poutcome','day_of_week'], axis=1, inplace=True)
        df_test['housemaid'] = pd.Series(newjob[:,0], index=df_test.index)
        df_test['services'] = pd.Series(newjob[:,1], index=df_test.index)
        df_test['admin.'] = pd.Series(newjob[:,2], index=df_test.index)
        df_test['blue-collar'] = pd.Series(newjob[:,3], index=df_test.index)
        df_test['technician'] = pd.Series(newjob[:,4], index=df_test.index)
        df_test['retired'] = pd.Series(newjob[:,5], index=df_test.index)
        df_test['management'] = pd.Series(newjob[:,6], index=df_test.index)
        df_test['unemployed'] = pd.Series(newjob[:,7], index=df_test.index)
        df_test['self-employed'] = pd.Series(newjob[:,8], index=df_test.index)
        df_test['job_unknown'] = pd.Series(newjob[:,9], index=df_test.index)
        df_test['entrepreneur'] = pd.Series(newjob[:,10], index=df_test.index)
        df_test['student'] = pd.Series(newjob[:,11], index=df_test.index)

        df_test['married'] = pd.Series(newmarital[:,0], index=df_test.index)
        df_test['single'] = pd.Series(newmarital[:,1], index=df_test.index)
        df_test['divorced'] = pd.Series(newmarital[:,2], index=df_test.index)
        df_test['marital_unknown'] = pd.Series(newmarital[:,3], index=df_test.index)

        df_test['basic_4y'] = pd.Series(newedu[:,0], index=df_test.index)
        df_test['high.school'] = pd.Series(newedu[:,1], index=df_test.index)
        df_test['basic_6y'] = pd.Series(newedu[:,2], index=df_test.index)
        df_test['basic_9y'] = pd.Series(newedu[:,3], index=df_test.index)
        df_test['professional_course'] = pd.Series(newedu[:,4], index=df_test.index)
        df_test['education_unknown'] = pd.Series(newedu[:,5], index=df_test.index)
        df_test['university_degree'] = pd.Series(newedu[:,6], index=df_test.index)
        df_test['illiterate'] = pd.Series(newedu[:,7], index=df_test.index)

        df_test['default_no'] = pd.Series(newdefa[:,0], index=df_test.index)
        df_test['default_unknown'] = pd.Series(newdefa[:,1], index=df_test.index)
        df_test['default_yes'] = pd.Series(newdefa[:,2], index=df_test.index)

        df_test['housing_no'] = pd.Series(newhous[:,0], index=df_test.index)
        df_test['housing_yes'] = pd.Series(newhous[:,1], index=df_test.index)
        df_test['housing_unknown'] = pd.Series(newhous[:,2], index=df_test.index)

        df_test['loan_no'] = pd.Series(newloan[:,0], index=df_test.index)
        df_test['loan_yes'] = pd.Series(newloan[:,1], index=df_test.index)
        df_test['loan_unknown'] = pd.Series(newloan[:,2], index=df_test.index)

        df_test['telephone'] = pd.Series(newcontact[:,0], index=df_test.index)
        df_test['cellular'] = pd.Series(newcontact[:,1], index=df_test.index)


        df_test['mar'] = pd.Series(newmonth[:,0], index=df_test.index)
        df_test['apr'] = pd.Series(newmonth[:,1], index=df_test.index)
        df_test['may'] = pd.Series(newmonth[:,2], index=df_test.index)
        df_test['jun'] = pd.Series(newmonth[:,3], index=df_test.index)
        df_test['jul'] = pd.Series(newmonth[:,4], index=df_test.index)
        df_test['aug'] = pd.Series(newmonth[:,5], index=df_test.index)
        df_test['sep'] = pd.Series(newmonth[:,6], index=df_test.index)
        df_test['oct'] = pd.Series(newmonth[:,7], index=df_test.index)
        df_test['nov'] = pd.Series(newmonth[:,8], index=df_test.index)
        df_test['dec'] = pd.Series(newmonth[:,9], index=df_test.index)

        df_test['mon'] = pd.Series(newDay[:,0], index=df_test.index)
        df_test['tue'] = pd.Series(newDay[:,1], index=df_test.index)
        df_test['wed'] = pd.Series(newDay[:,2], index=df_test.index)
        df_test['thu'] = pd.Series(newDay[:,3], index=df_test.index)
        df_test['fri'] = pd.Series(newDay[:,4], index=df_test.index)

        df_test['nonexistent'] = pd.Series(newpout[:,0], index=df_test.index)
        df_test['failure'] = pd.Series(newpout[:,1], index=df_test.index)
        df_test['success'] = pd.Series(newpout[:,2], index=df_test.index)
        
        df_test['jan'] = np.nan
        df_test['feb'] = np.nan
        df_test=df_test.fillna(0)
        
        arrange = df_test.columns.tolist()
        arrange.remove('result')
        arrange.append('result')
        df_test = df_test[arrange]

        df_test.to_csv("LuigiFETest.csv",index=False)
        df.to_csv(self.output().path,index=False)

class FeatureSelection(luigi.Task):
    
    def requires(self):
        yield FeatureEngineering()
    
    def output(self):
        return luigi.LocalTarget("LuigiFS.csv")
    
    def run(self):
        df=pd.read_csv(FeatureEngineering().output().path)
        df_test = pd.read_csv("LuigiFETest.csv")
        # df_test =df_test.drop('Unnamed: 0',axis=1)
        # df=df.drop('Unnamed: 0',axis=1)
        X_train= df.drop(['result'], axis=1)
        y_train= df['result']
        X_test= df_test.drop(['result'], axis=1)
        y_test= df_test['result']

        sgdc = SGDClassifier()
        rfe_1 = RFECV(sgdc,n_jobs=2,verbose=1,cv=10,step=5)
        rfe_1= rfe_1.fit(X_train, y_train)
        rfe_1.ranking_
        selected_columns = ['age', 'campaign', 'pdays', 'previous', 'emp_var_rate','nr_employed','housemaid','cons_price_idx', 'cons_conf_idx', 'euribor3m', 'result']
        new_df_train = df[selected_columns]
        new_df_test = df_test[selected_columns]
        # X_train = df.iloc[:,:len(df.columns)-1]
        # y_train = df.iloc[:,len(df.columns)-1]
        new_df_test.to_csv("LuigiFSTest.csv", index=False)
        new_df_train.to_csv(self.output().path, index = False)

class PredictionModel(luigi.Task):

    aws_access_key_id = luigi.Parameter()
    aws_secret_access_key = luigi.Parameter()

    def requires(self):
        yield FeatureSelection()
    
    def output(self):
        return print("Success")
    
    def run(self):
        df=pd.read_csv(FeatureSelection().output().path)
        df_test=pd.read_csv("LuigiFSTest.csv")
        X_train= df.drop(['result'], axis=1)
        y_train= df['result']
        X_test= df_test.drop(['result'], axis=1)
        y_test= df_test['result']

        logistic=LogisticRegression(max_iter = 50,n_jobs = 2,penalty ='l2')
        logistic.fit(X_train,y_train)
        predict_test = logistic.predict(X_test)
        report1 = classification_report(y_test, predict_test)

        sgdc = SGDClassifier(alpha=0.5,max_iter=50, n_jobs=4, penalty='elasticnet', power_t=0.5, random_state=None, shuffle=True, tol=None, verbose=0, warm_start=False)
        sgdc.fit(X_train,y_train)
        predict_sgdc = sgdc.predict(X_test)
        report2 = classification_report(y_test, predict_sgdc)

        mlp = MLPClassifier(activation ='identity',alpha = 1,hidden_layer_sizes=(50,50,50),max_iter=200)
        mlp.fit(X_train,y_train)
        predict_mlp= mlp.predict(X_test)
        report3 = classification_report(y_test, predict_mlp)

        def classifaction_report_csv(report):
            report_data = []
            lines = report.split('\n')
            for line in lines[2:-3]:
                row = {}
                print("GAURANG")
                print(line)
                row_data = line.split('      ')
                row['class'] = row_data[0]
                row['precision'] = float(row_data[1])
                row['recall'] = float(row_data[2])
                row['f1_score'] = float(row_data[3])
                row['support'] = float(row_data[4])
                report_data.append(row)
            dataframe = pd.DataFrame.from_dict(report_data)
            return dataframe

        report1 = classifaction_report_csv(report1)
        report2 = classifaction_report_csv(report2)
        report3 = classifaction_report_csv(report3)

        filename = 'SGD_Classifier.pkl'
        pickle.dump(sgdc, open(filename, 'wb'))
        filename = 'MLP_Classifier.pkl'
        pickle.dump(mlp, open(filename, 'wb'))
        filename = 'logistic_clf.pkl'
        pickle.dump(logistic, open(filename, 'wb'))

        report1.to_csv("LogisticReport.csv", index=False)
        report2.to_csv("SGDReport.csv", index=False)
        report3.to_csv("MLPReport.csv", index=False)

        readable = datetime.datetime.fromtimestamp(time.time()).isoformat()
        # s3 = boto3.resource('s3')
        
        # session = boto3.session.Session(region_name='us-east-1')
        # s3 = session.client('s3', config= boto3.session.Config(signature_version='s3v4'),aws_access_key_id=self.aws_access_key_id,
        #  aws_secret_access_key=self.aws_secret_access_key)
        conn = S3Connection(self.aws_access_key_id, self.aws_secret_access_key)
        b = conn.get_bucket('bankdepositterm')
        k = Key(b)
        k.key = 'LogisticReport.csv'
        k.set_contents_from_filename('LogisticReport.csv')
        k.set_acl('public-read')
        k.key = 'SGDReport.csv'
        k.set_contents_from_filename('SGDReport.csv')
        k.set_acl('public-read')
        k.key = 'MLPReport.csv'
        k.set_contents_from_filename('MLPReport.csv')
        k.set_acl('public-read')
        k.key = 'MLP_Classifier.pkl'
        k.set_contents_from_filename('MLP_Classifier.pkl')
        k.set_acl('public-read')
        k.key = 'logistic_clf.pkl'
        k.set_contents_from_filename('logistic_clf.pkl')
        k.set_acl('public-read')
        k.key = 'SGD_Classifier.pkl'
        k.set_contents_from_filename('SGD_Classifier.pkl')
        k.set_acl('public-read')
        # s3.create_bucket(Bucket='bankdepositterm'+readable)
        # data = open('LogisticReport.csv', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='LogisticReport.csv', Body=data)
        # data = open('SGDReport.csv', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='SGDReport.csv', Body=data)
        # data = open('MLPReport.csv', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='MLPReport.csv', Body=data)
        # data = open('MLP_Classifier.pkl', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='MLP_Classifier.pkl', Body=data)
        # data = open('logistic_clf.pkl', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='logistic_clf.pkl', Body=data)
        # data = open('SGD_Classifier.pkl', 'rb')
        # s3.put_object(Bucket='bankdepositterm', Key='SGD_Classifier.pkl', Body=data)

        self.output()

if __name__ == '__main__':
    luigi.run()