FROM ubuntu:16.04
FROM python:3

RUN pip install pandas
RUN pip install Flask
RUN pip install boto
RUN pip install boto3
RUN pip install sklearn
RUN pip install scipy

#Create static directory

RUN mkdir -p /src/assignment2/static
WORKDIR /src/assignment2

#Add Required files

COPY static/* /src/assignment2/static/

#Create templates directory
RUN mkdir -p /src/assignment2/templates

#Add Required files

COPY templates/* /src/assignment2/templates/

COPY app.py /src/assignment2/

EXPOSE 8000
#CMD [ “python ”, “./app.py” ]
CMD ["sh", "-c", "python /src/assignment2/app.py --aws-access-key-id ${ak} --aws-secret-access-key ${sk} ", "-p 8000"]
#CMD ["python", "/src/simpleapp.py", "-p 8000"]