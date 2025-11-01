#!/bin/bash

nohup python train.py --model ResNet --epochs 50 --batch_size 512 > train1.log 2>&1 &
wait

nohup python test.py --model ResNet > test1.log 2>&1 &