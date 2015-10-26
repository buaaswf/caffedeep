#!/usr/bin/env python
# encoding: utf-8
import re
def mklabel(csv):
    with open(csv, 'r') as f and open("dst.csv", 'w') as wf:
        pattern = re.compile(u'/lfw-deepfunneled_align/(.*?)/')
        for line in f:
            label = pattern.findall(line.strip())
            new_line = line.strip() + ' ' + label[0] + '\n'
            wf.write(new_line)
mklabel("file.csv")

