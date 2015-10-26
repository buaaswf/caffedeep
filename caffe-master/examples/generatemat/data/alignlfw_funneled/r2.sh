#!/bin/bash
count=0
#total=`ls -l |wc -l`
for img in `ls`
do
    if [ -d $img ];then
        new=$count
        mv "$img" "$new"
        #if [$? -eq 0 ];then
        #echo "Renaming $img to $new"
        let count++
        #fi
    fi
done
