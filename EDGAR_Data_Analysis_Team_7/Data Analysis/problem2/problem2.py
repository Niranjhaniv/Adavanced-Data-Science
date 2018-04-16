import csv
import os
import zipfile
import pandas as pd
import numpy as np
import glob
import sys
import logging
from bs4 import BeautifulSoup
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import time
import datetime


def summary(all_data):
    data = pd.DataFrame()
    data = all_data
    data.info()
    # describe the data that we have 
    data.describe()
    # summary = pd.DataFrame()
    #logging.debug('In the function : summary')
    csvpath=str(os.getcwd())
   
    # Create a new column TimeStamo for analysis purpose
    data['Timestamp'] = data[['date', 'time']].astype(str).sum(axis=1)
   
    # Create a summary that groups ip by date
    ipsummary=data['ip'].groupby(data['date']).describe()
    summaryipdescribe = pd.DataFrame(ipsummary)
    s=summaryipdescribe.transpose()
    s.to_csv(csvpath+"/ipsummarybydatedescribe.csv")
   
    # get Top 10 count of all cik with their accession number
    data['COUNT'] = 1  # initially, set that counter to 1.
    group_data = data.groupby(['date', 'cik', 'accession'])['COUNT'].count()  # sum function
    rankedData=group_data.rank()
    summarygroup=pd.DataFrame(rankedData)
    summarygroup.to_csv(csvpath+"/Top10cik.csv")
   
    # For anomaly detection -check the length of cik
    data['cik'] = data['cik'].astype('str')
    data['cik_length'] = data['cik'].str.len()
    data[(data['cik_length'] > 10)]
    data['COUNT'] = 1
    datagroup=pd.DataFrame(data)
    datagroup.to_csv(csvpath+"/LengthOfCikForAnomalyDetection.csv")
 
    # Create a summary that groups cik by accession number
    summary2 = data['extention'].groupby(data['cik']).describe()
    summarycikdescribe = pd.DataFrame(summary2)
    summarycikdescribe.to_csv(csvpath+"/summarycikbyextentiondescribe.csv")

def replace_missingValues(df):
        df.shape
        df.isnull().sum()
        df.info()
        df.dropna(subset = ['cik','ip','accession','date','time'])
        # zone
        max_no_agent=pd.DataFrame(df.groupby('zone').size().rename('cnt')).idxmax()[0]
        df= df.fillna({'zone':max_no_agent})
        #browser
        df['browser']=df['browser'].fillna('missing')
        
        #extension
        df.loc[df["extention"].str.startswith("."), "extention"]=df['accession'].map(str) + df['extention']
        #code
        df['code'] = df['code'].fillna('Unknown_status')
        #size
        df['size'] = df['size'].fillna(0)
        #idx
        df.loc[df['idx'].isin([0.0,1.0])]
        max_no_agent=pd.DataFrame(df.groupby('idx').size().rename('cnt')).idxmax()[0]
        df=df.fillna({'idx':max_no_agent})
        #norefer
        df.loc[df['norefer'].isin([0.0,1.0])]
        max_no_agent=pd.DataFrame(df.groupby('norefer').size().rename('cnt')).idxmax()[0]
        df['norefer']=df['norefer'].fillna(max_no_agent)
        #noagent
        df.loc[df['noagent'].isin([0.0,1.0])]
        max_no_agent=pd.DataFrame(df.groupby('noagent').size().rename('cnt')).idxmax()[0]
        df['noagent']=df['noagent'].fillna(max_no_agent)
        #find
        df['find'].between(0.0,10.0).all()
        df['find']=df['find'].fillna(max_no_agent)
        #crawler
        df['crawler'] = df['crawler'].fillna(0)
      
        return df


def change_dataTypes(df):
    logging.debug('In the function : change_dataTypes')
    df = replace_missingValues(df)
    df['cik'] = df['cik'].astype('int64')
    df['code'] = df['code'].astype('object')
    df['size'] = df['size'].astype('int64')
    df['idx'] = df['idx'].astype('int64')
    df['norefer'] = df['norefer'].astype('int64')
    df['noagent'] = df['noagent'].astype('int64')
    df['crawler'] = df['crawler'].astype('int64')
    df.to_csv("merged.csv",encoding='utf-8')
    summary(df)
    return 0


def create_dataframe(path):
    logging.debug('In the function : create_dataframe')
    df = pd.DataFrame()
    allFiles = glob.glob(path + '/log*.csv')
    for file_ in allFiles:
        df2 = pd.read_csv(file_,index_col=None, header=0) 
        df=df.append(df2,ignore_index=True)
        logging.debug("check")
        del df2
    print(df)
    change_dataTypes(df)
    return df


def assure_path_exists(path):
    logging.debug('In a function : assure_path_exists')
    if not os.path.exists(path):
        os.makedirs(path)


def get_dataOnLocal(monthlistdata, year):
    logging.debug('In the function : get_dataOnLocal')
    df = pd.DataFrame()
    foldername = str(year)
    path = str(os.getcwd()) + "/" + foldername
    assure_path_exists(path)
    for month in monthlistdata:
        with urlopen(month) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(path)
    create_dataframe(path)
    return 0


def get_allmonth_data(linkhtml, year):
    logging.debug('In the function : get_allmonth_data')
    allzipfiles = BeautifulSoup(linkhtml, "html.parser")
    ziplist = allzipfiles.find_all('li')
    monthlistdata = []
    count = 0
    for li in ziplist:
        zipatags = li.findAll('a')
        for zipa in zipatags:
            if "01.zip" in zipa.text:
                monthlistdata.append(zipa.get('href'))
    get_dataOnLocal(monthlistdata, year)



def get_url(year):
    logging.debug('In the function : get_url')
    page = urlopen('https://www.sec.gov/dera/data/edgar-log-file-data-set.html')
    soup=BeautifulSoup(page ,'html.parser')

    all_div = soup.findAll("div", attrs={'id': 'asyncAccordion'})
    logging.debug(soup)
    for div in all_div:
        h2tag = div.findAll("a")
        for a in h2tag:
            if str(year) in a.get('href'):
                year_link= a.get('href')
    com_year_link = "https://www.sec.gov"+year_link
    logging.debug(com_year_link)
    year_link_page=urlopen(com_year_link)
    logging.debug('before all month')
    get_allmonth_data(year_link_page, year)


def valid_year(year):
    logging.debug('In the function : valid_year')
    logYear = ['2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015','2016','2017','2018']
    for log in logYear:
        try:
            if year in log:
                get_url(year)
        except:
            print("Data for" + year + "does not exist")
            "Data for" + year + "does not exist"


def main():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    args = sys.argv[1:]

    year = ''
    counter = 0
    if len(args) == 0:
        year = "2003"
    for arg in args:
        if counter == 0:
            year= str(arg)
        counter += 1
    logfilename = 'log_Edgar_'+ year + '_' + st + '.txt'
    logging.basicConfig(filename=logfilename, level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Program Start')
    logging.debug('*************')    
    logging.debug('Calling the initial URL'.format(year))
    valid_year(year)


if __name__ == '__main__':
    main()

