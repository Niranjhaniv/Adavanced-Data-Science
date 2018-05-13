FROM ubuntu:16.04
FROM python:3

RUN pip install Flask
RUN pip install django
RUN pip install boto3
RUN pip install tensorflow
RUN pip install keras
#RUN pip install openCV
RUN pip install instalooter
#RUN pip install tarfile
RUN pip install numpy
RUN pip install pillow
RUN pip install opencv-python
RUN pip install s3fs

#Create static directory
RUN mkdir -p /src/static
WORKDIR /src

#Copy Required files

COPY static/* /src/static/

#Create templates directory
RUN mkdir -p /src/templates
RUN mkdir -p /src/Flask

#Copy Required files

COPY templates/* /src/templates/
COPY Flask/* /src/Flask/
COPY app.py /src/

EXPOSE 5000
CMD python app.py