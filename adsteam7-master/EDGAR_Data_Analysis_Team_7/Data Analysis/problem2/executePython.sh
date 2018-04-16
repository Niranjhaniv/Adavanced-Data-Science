#!/bin/bash

python3.5 /src/hw1_part2/problem2.py $1

mv /src/hw1_part2/*.csv /src/hw1_part2/generatedFiles
mv /src/hw1_part2/log*.txt /src/hw1_part2/generatedFiles

if [ $? -eq 0 ]
then
  echo "Successfully created file"
  #sh /src/hw1_part2/awsS3.sh $1 $2 $3
else
  echo "Could not create file" >&2
fi
