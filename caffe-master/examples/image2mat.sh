#!/bin/bash
cd generatemat &&
cd data
cd $1
#cp ../r2.sh .
#./r2.sh &&
#cd ..
find `pwd` -name "*.jpg" > ${1}file.txt &&
cd ..
cd ..
echo `pwd`
python mklabelforimages.py $1
cd ..
python feature_test.py generatemat/data/$1 generatemat/${1}.mat ./generatemat/${1}dst.csv 

