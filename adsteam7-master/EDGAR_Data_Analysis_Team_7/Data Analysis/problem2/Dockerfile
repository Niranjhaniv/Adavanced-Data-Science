FROM ubuntu:16.04
USER root

# Install dependencies
RUN apt-get update
RUN apt-get install -y python-pip --upgrade python-pip
RUN pip install --upgrade pip

# install py3
RUN apt-get update -q
RUN apt-get install -y python3.5 python3.5-dev python3-pip python3-setuptools pkg-config

RUN pip3 install --upgrade pip

# install additional python packages
RUN pip3 install ipython
RUN pip3 install BeautifulSoup4
RUN pip3 install scipy
RUN pip3 install pandas
RUN pip3 install numpy

#install AWS CLI
RUN pip3 install awscli

WORKDIR /src/
RUN mkdir -p /src/hw1_part2/generatedFiles
ADD executePython.sh /src/hw1_part2
ADD awsS3.sh /src/hw1_part2
ADD problem2.py /src/hw1_part2
RUN chmod +x /src/hw1_part2/executePython.sh
RUN chmod +x /src/hw1_part2/awsS3.sh