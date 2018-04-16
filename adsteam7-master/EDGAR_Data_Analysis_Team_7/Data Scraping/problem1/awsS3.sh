#!/bin/bash

AWS_ACCESS=$1
AWS_SECRET=$2
AWS_S3=$3

export ACCESS_KEY=${AWS_ACCESS}
export SECRET_KEY=${AWS_SECRET}
export S3_PATH=${AWS_S3}

set -e

: ${ACCESS_KEY:?"ACCESS_KEY env variable is required"}
: ${SECRET_KEY:?"SECRET_KEY env variable is required"}
: ${S3_PATH:?"S3_PATH env variable is required"}
export DATA_PATH=${DATA_PATH:-/data/}

echo "access_key=$ACCESS_KEY"
echo "secret_key=$SECRET_KEY"
echo "S3_PATH=$S3_PATH"

aws configure set aws_access_key_id $ACCESS_KEY
aws configure set aws_secret_access_key $SECRET_KEY
aws configure set default.region us-east-2

echo "AWS S3 Job started: $(date)"

aws s3 sync /src/hw1_part1/generatedFiles  $S3_PATH 

echo "AWS S3 Job finished: $(date)"
