FROM ubuntu:16.04
FROM python:3

RUN mkdir -p /src
WORKDIR /src

RUN pip install pandas
RUN pip install boto3
RUN pip install numpy
RUN pip install luigi
RUN pip install sklearn
RUN pip install scipy
RUN pip install tensorflow
RUN pip install keras
RUN pip install pillow
RUN pip install h5py

COPY pipeline.py /src/

CMD ["luigid","&"]

CMD [ "python /src/pipeline.py","BuildKeras","--local-scheduler","--accessKey","ASDSCSDFWEF","--secretAccessKey","ASDADADASD"]