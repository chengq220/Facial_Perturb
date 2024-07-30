#!/bin/bash -f

cd /home/ps/qiuc/emodec/
tdyDate=$(date +%m%d%y)
nohup python /home/ps/qiuc/emodec/train.py > /home/ps/qiuc/emodec/log/log_${tdyDate}_wb3.log 2>&1 &
