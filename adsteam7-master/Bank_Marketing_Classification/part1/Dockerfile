FROM ubuntu:16.04
FROM python:3

RUN mkdir -p /src/assignment2
WORKDIR /src/assignment2

RUN pip install pandas
RUN pip install boto3
RUN pip install numpy
RUN pip install boto3
RUN pip install urllib3
RUN pip install luigi
RUN pip install sklearn
RUN pip install scipy

COPY pipeline.py /src/assignment2/

CMD ["sh", "-c", "python /src/assignment2/pipeline.py PredictionModel --local-scheduler --aws-access-key-id ${ak} --aws-secret-access-key ${sk} "]