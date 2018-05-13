<h1> Brand Logo Visibility using Convolutional Neural Network </h1>

The advertising and branding industry has become more crowded than it previously was and it is not just increasing but increasing exponentially which is a major cause of concern for all those who also want their branding and advertising campaigns to reap expected benefits for them. Every company logo is the first and most important branding tool and it has the power to effectively register an overhaul in your company’s recognition and success. Logo is what defines brand and conveys the ideology and purpose of the brand to the consumer. If it’s instantly recognizable, then rest assured, the sales will look north. Our project recognises logo present in any given image. The image logo classification was built in two different models of neural network for comparison. Later, the project was deployed in EC2 instance to create web application with user interface.


<h2> Inception V3 </h2>

Modern image recognition models have millions of parameters. Training them from scratch requires a lot of labelled training data and a lot of computing power (hundreds of GPU-hours or more). Transfer learning is a technique that shortcuts much of this by taking a piece of a model that has already been trained on a related task and reusing it in a new model. We are using retrain.py from below github:
https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py

retrain.py – This python file consists of the retraining model that creates the last layer of the convolutional neural network with latest image classification

lable_image.py – This python file helps in testing our classification accuracy

<h2> ModelImplementation </h2>

The folder consists of the convolutional neural network build using keras with tensorflow as backend. The keras model was pipelined and dockerised for running the model irrelevant of the environment and package dependencies.

<h2> LogoDtection </h2>

Contains the flask required templates and python files

<h2> EC2 instance containg web application </h2>
ec2-18-233-167-186.compute-1.amazonaws.com



