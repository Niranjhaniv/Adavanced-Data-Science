from flask import Flask
from flask import Flask, flash, redirect, render_template, request, url_for
import os
import json
from django.core.files.temp import NamedTemporaryFile
import base64
import sys
import tempfile
from keras.optimizers import SGD
import cv2
from keras.models import load_model
import os.path
import boto3
from boto3.session import Session
import cv2
import re
import sys
import tarfile
from instalooter import looters
from PIL import Image
import glob
import io

import numpy as np
from six.moves import urllib
import tensorflow as tf
from base64 import b64decode

# !flask/bin/python
import s3fs
from flask import Flask, jsonify
from flask import make_response
from flask import request, render_template
# from flask_bootstrap import Bootstrap
from werkzeug import secure_filename
fs = s3fs.S3FileSystem(anon=False)


import sys, os

app = Flask(__name__)

UPLOAD_FOLDER = 'Flask/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

session = boto3.session.Session(region_name='us-east-1')
s3client = session.client('s3', config= boto3.session.Config(signature_version='s3v4'),aws_access_key_id='<accesskey>',
         aws_secret_access_key='<aws_secret_access_key>')

model = load_model('Flask/weights.best.hdf5')

def allowed_file(filename):
    # this has changed from the original example because the original did not work for me
    return filename[-3:].lower() in ALLOWED_EXTENSIONS


def s3_download():
    bucketName = "data-brand-logos"
    session = Session(aws_access_key_id='AKIAIFICJKDVRYOIBOFQ',
                      aws_secret_access_key='I4g18x1t2RsN6R2XIMjhGNMum7oLA546Qsjex+0D')
    s3 = session.resource('s3')

    your_bucket = s3.Bucket(bucketName)
    KEY1 = 'output_graph.pb'
    KEY2  = 'output_labels.txt'
    your_bucket.download_file(KEY1, "Flask/output_graph.pb")
    your_bucket.download_file(KEY2, "Flask/output_labels.txt")

def hash_tag(tag_word,no_of_images):
    looter = looters.HashtagLooter(tag_word)
    return looter.download_pictures(os.getcwd()+"hash_images/"+tag_word, media_count=no_of_images)
def public_user(user,no_of_user_images):
    user = looters.ProfileLooter(user)
    return user.download_pictures(os.getcwd()+"user_image/", media_count=no_of_user_images)




@app.route('/')
def home():
    return render_template('login.html');


# import default command line flags from TensorFlow
FLAGS = tf.app.flags.FLAGS

# define directory that the model is stored in (default is the current directory)
tf.app.flags.DEFINE_string(
    'model_dir', '',
    """Flask/output_graph.pb, """
    """Flask/output_labels.txt""")

tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


@app.route('/login', methods=['POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == 'password' and request.form.get('username') == 'admin':
            return redirect(url_for('main', access="false"))
        if request.form.get('password') == 'password' and request.form.get('username') == 'user':
            return redirect(url_for('main', access="true"))

        else:
            return redirect(url_for(''))


@app.route("/logout")
def logout():
    return "Hello World!"


@app.route("/register")
def register():
    return render_template('register.html');



@app.route('/predict', methods=['POST'])
def predict_image():
    classes_array =['Adidas','Apple','BMW','Citroen','Cocacola','DHL','Fedex','Ferrari','Ford','Google','Heineken','HP','Intel','McDonalds',
    'Mini','Nbc','Nike','Pepsi','Porsche','Puma','RedBull','Sprite','Starbucks','Texaco','Unicef','Vodafone','Yahoo']
    opt = SGD(momentum=0.9, lr=5e-3)
    model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
    # Preprocess the image so that it matches the training input
    img = Image.open(request.files['file'])
    img = np.array(img)
    img = cv2.resize(img, (64, 64))

    img = np.reshape(img, [1, 64, 64, 3])
    # Use the loaded model to generate a prediction.
    pred = model.predict(img)
    digit = np.argmax(pred)

    prediction = {'logo': classes_array[digit]}
    return render_template('output.html', data_json=prediction);



@app.route('/insta', methods=['POST'])
def insta():
    if request.method == 'POST':
        if request.form.get('username') != '':
            username = request.form.get('username')
            public_user(username, 3)
        if request.form.get('hashtag') != '':
            hashtag = request.form.get('hashtag')
            hash_tag(hashtag, 3)
    return render_template('insta.html');


@app.route("/classify", methods=["POST"])
def classify():

    if request.method == 'POST':
        #s3_download()
        data = {"success": False}
        sample_data = {"success": False}
        tmp_f = NamedTemporaryFile()
        image_path_list = []
        for f in request.files.getlist('files'):
            #image_path_list.append(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            #f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
            image_bytes = f.read()
            image = Image.open(io.BytesIO(image_bytes))
            image.save(tmp_f, image.format)
        # for imagePath in image_path_list:
        #     image = cv2.imread(imagePath)

            create_graph()
            print("Model loaded")

            node_lookup = NodeLookup()
            print("Node lookup loaded")

            predictions = dict(run_inference_on_image(tmp_f))
            #print(predictions)
            data["predictions"] = {}
            s1 = json.dumps(predictions)
            data_json=json.loads(s1)
            data["predictions"]=data_json
            sample_data["confidence"]=data_json

            # for category in data['predictions'].copy():
            #      percent = data["predictions"][category] * 100;
            #      sample_data["confidence"][category]["percent"]=percent
            # print(sample_data)
            return render_template('classify.html', data_json=data);
    return ""


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    print("node created");

    def __init__(self, label_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'Flask/output_labels.txt')

        self.node_lookup = self.load(label_lookup_path)

    def load(self, label_lookup_path):
        """Loads a human readable English name for each softmax node.
        Args:
          label_lookup_path: string UID to integer node ID.
        Returns:
          dict from integer node ID to human-readable string.
        """

        node_id_to_name = {}

        label_file = open(label_lookup_path)
        i = 0

        # labels are ordered from 0 to N in the lookup file

        for line in label_file:
            node_id_to_name[i] = line.strip()
            i = i + 1

        return node_id_to_name

    # return the friendly name for the given node_id
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def create_graph():
    print("Graph created");
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'Flask/output_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def get_image_list(path):
    return glob.glob(path + '*.jpg')


# sess = None
# node_lookup = None


def run_inference_on_image(image_file):
    """Runs inference on an image.
    Args:
      image_data: Image data.
    Returns:
      Nothing
    """
    sess = tf.Session()
    print("Tensorflow session ready")
    node_lookup = NodeLookup()
    image_data = tf.gfile.FastGFile(image_file.name, 'rb').read()
    print("Node lookup loaded")
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # sort the predictions
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    # map to the friendly names and return the tuples
    return [(node_lookup.id_to_string(node_id), float(predictions[node_id])) for node_id in top_k]


@app.route("/main")
def main():
    access = request.args.get('access', None)

    return render_template('main.html', access=access);


if __name__ == "__main__":
	app.run(host='0.0.0.0',port=5000,debug=True)
