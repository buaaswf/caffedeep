#!/bin/bash

#!/usr/bin/env sh
# args for EXTRACT_FEATURE
TOOL=./build/tools
MODEL=models/idface/deepid_28_0_train_iter_90000.caffemodel #下载得到的caffe model
PROTOTXT=examples/_temp/patch_train_original.prototxt # 网络定义
LAYER=fc6 # 提取层的名字，如提取fc7等
LEVELDB=examples/_temp/original/feat # 保存的leveldb路径:
BATCHSIZE=10

# args for LEVELDB to MAT
DIM=160 # 需要手工计算feature长度
OUT=examples/_temp/features.mat #.mat文件保存路径
BATCHNUM=224 # 有多少哥batch， 本例只有两张图， 所以只有一个batch

$TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE leveldb GPU 0 
echo $TOOL/extract_features.bin  $MODEL $PROTOTXT $LAYER $LEVELDB $BATCHSIZE leveldb GPU 0
ipython leveldb2mat.py $LEVELDB $BATCHNUM  $BATCHSIZE $DIM $OUT 
