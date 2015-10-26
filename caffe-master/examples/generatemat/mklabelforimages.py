#!/usr/bin/env python
# encoding: utf-8
import re
import sys
def mklabel(csv,dst):
    with open(csv, 'r') as f , open(dst+'dst.csv', 'w') as wf:
        pattern = re.compile(u'/'+dst+'/(.*?)/')
        print  pattern.pattern
        for line in f:
            label = pattern.findall(line.strip())
            new_line = line.strip() + ' ' + label[0] + '\n'
            wf.write(new_line)
#mklabel("./data/"+sys.argv[1]+"file.txt",sys.argv[1])
mklabel("./data/imagelist_lfw.txt",sys.argv[1])
