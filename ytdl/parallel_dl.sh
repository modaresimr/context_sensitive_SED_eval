#!/bin/bash
if [ $# -lt 2 ];then
   printf "Not enough arguments - \n"
   printf "Usage: parallel_dl.sh arg1 arg2 [arg3]"
   printf "arg1: the dataset file in tsv format"
   printf "arg2: output directory"
   printf "arg3: optional: you can filter spesific class labels using regular expression"
   printf "e.g., parallel_dl.sh eval.tsv data"
   printf "This will download 10 second audio of files in that dataset and save it in data folder
   exit 0
fi

TSV=$1
DIR=$2
[ ! -d $DIR ] && mkdir -p $DIR
REGX=$3 || "/m/07yv9|/m/032s66"
[] mkdir -p $DIR
egrep $TSV -i "$REGX"| cut -f1 | awk '!a[$0]++'|parallel --eta   -j 8 "echo download $DIR {1} ; bash download.sh {1} $DIR"
