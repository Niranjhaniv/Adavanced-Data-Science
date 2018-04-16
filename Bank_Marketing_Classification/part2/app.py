from flask import Flask, render_template, redirect,request,Response,url_for, send_from_directory,make_response,send_file
import pandas as pd
import os
import json
import csv
import s3fs
import boto3
import boto3.session
import requests
import logging
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask.ext.mysqldb import MySQL
import pickle
from flask import Flask
from flask import request
#from werkzeug import secure_filename
from flask import jsonify

# from flask import jsonify
# from flask import abort


from werkzeug.utils import secure_filename
fs = s3fs.S3FileSystem(anon=False)




UPLOAD_FOLDER = 'flask/uploads'
ALLOWED_EXTENSIONS = set(['csv'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MYSQL_HOST'] = 'sql9.freemysqlhosting.net'
app.config['MYSQL_USER'] = 'sql9232325'
app.config['MYSQL_PASSWORD'] = 'anNkiUumJ7'
app.config['MYSQL_DB'] = 'sql9232325'
mysql = MySQL(app)


session = boto3.session.Session(region_name='us-east-1')
s3client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'),aws_access_key_id='AKIAI53GFRQDZBQGWJ5A',
         aws_secret_access_key='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')





def allowed_file(filename):
    print('.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS)
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
@app.route("/")
def index():
    cur = mysql.connection.cursor()
    cur.execute('''SELECT * FROM loginCredentials ''')
    rv = cur.fetchall()
    return str(rv)


# class LoginForm(FlaskForm):
#     StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
#     PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
#     remember = BooleanField('remember me')


app.config['SECRET_KEY'] = 'DontTellAnyone'

# @app.route('/login')
# def login():
#         username = request.form['username']
#         return render_template('layout.html', age=age)
#     return render_template('login.html')
#


def load_pickle(model):
    if model == 'MLP':
        response = s3client.get_object(Bucket='bankdepositterm', Key='MLP_Classifier.pkl')
        pickled_list = response['Body'].read()
        pickleVal = pickle.loads(pickled_list)
        return pickleVal
    elif model == 'MLP':
            response = s3client.get_object(Bucket='bankdepositterm', Key='SGD_Classifier.pkl logistic_reg.pkl')
            pickled_list = response['Body'].read()
            pickleVal = pickle.loads(pickled_list)
            return pickleVal
    elif model == 'MLP':
        response = s3client.get_object(Bucket='bankdepositterm', Key='logistic_reg.pkl')
        pickled_list = response['Body'].read()
        pickleVal = pickle.loads(pickled_list)
        return pickleVal

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/uploadJson', methods=['GET', 'POST'])
def uploadJson():

    if request.method == 'POST':
        data = request.form.get("uploadtype")
        model = request.form.get("model")
        pickleSel=  load_pickle(model)
        if 'file' not in request.files:

            return Response('No file uploaded', status=500)

        f = request.files['file']
        if f.filename == '':

            return Response('No filename uploaded', status=500)

        if f and allowed_file(f.filename):
            absolute_file = os.path.abspath(UPLOAD_FOLDER + f.filename)

            filename = secure_filename(absolute_file)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            datajson=read_csv(os.path.join(app.config['UPLOAD_FOLDER']+"/"+filename))
            df=pd.read_json(datajson)
            if data == 'single':
                if len(df.index) == 1:
                    y_pred = pickleSel.predict(df)
                    pred = pd.DataFrame(y_pred, columns=['Predict'])
                    pred.Predict.replace((1, 0), ('yes', 'no'), inplace=True)
                    result = pd.concat([df, pred], axis=1)
                    response = make_response(result.to_csv())
                    cd = 'attachment; filename=predicted_response.csv'
                    response.headers['Content-Disposition'] = cd
                    response.mimetype = 'text/csv'

                    return response


                else:
                    response ={
                            "error": {
                             "errors": [
                              {
                               "domain": "global",
                               "reason": "required",
                               "message": "Ohh no !! Csv contains more than 1 row. Use bulk option for more option ",
                               "locationType": "header",
                               "location": "Authorization"
                              }
                             ],
                             "code": 401,
                             "message": "Login Required"
                             }
                            }
                    return render_template('error.html', data=response)
            else:
                y_pred = pickleSel.predict(df)
                pred = pd.DataFrame(y_pred, columns=['Predict'])
                pred.Predict.replace((1, 0), ('yes', 'no'), inplace=True)
                result = pd.concat([df, pred], axis=1)
                response = make_response(result.to_csv())
                cd = 'attachment; filename=predicted_response.csv'
                response.headers['Content-Disposition'] = cd
                response.mimetype = 'text/csv'
                return response

def read_csv(file):
    csv_rows = []
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile)
        field = reader.fieldnames
        for row in reader:
            csv_rows.extend([{field[i]: row[field[i]] for i in range(len(field))}])
        datajson=write_json(csv_rows)
        return datajson

#Convert csv data into json and write it
def write_json(data):
    datajson=json.dumps(data)
    return datajson



if __name__ == '__main__':
    app.run(debug=True)
