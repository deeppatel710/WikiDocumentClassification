# -*- coding: UTF-8 -*-
import requests
import nltk
from bs4 import BeautifulSoup
import os

with open('fileid.txt', 'r') as f:
	files = f.read()
file_id = files.split()
 
corpus = []

for file in file_id:
	f = open(os.getcwd()+"/cleanFiles/"+file + '.csv' , 'r')
	doc = f.read().split(',')
	corpus.append(doc)
