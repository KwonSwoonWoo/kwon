# -*- coding: utf-8 -*- 
#example : python entry_tf_BIGDATA.py UTS
import sys
import os

file="tf_BIGDATA.py"
flag1='-i'
DB='BIGDATA.csv'
flag2='-T'
target = sys.argv[1]
#target_splited = "UTS/YS/El"
target_splited = "UTS/YS/J/El/HRc"
#target_splited = "UTS/YS/J/HRc"

os.system("python {} {} {} {} {} {}".format(file, flag1, DB, flag2,target,target_splited))


