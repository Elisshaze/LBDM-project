from numpy.lib.utils import source
import pandas as pd
from pandas import *
import glob
import os 
import plotly.graph_objects as go
from supervenn import supervenn
import matplotlib.pyplot as plt 
import re
import numpy as np
import seaborn as sns
import itertools
import sys, getopt

folder = "/home/elisa/LBDM-project/CTNNB1_csvs/*"

def retrieveName (f):
	base = f.split("/")[5].split(".")[0].split("@")[0]
	#base = "_".join(fname)
	return base


def readFile(fileName):
	fileObj = open(fileName, "r") #opens the file in read mode
	words = fileObj.read().splitlines() #puts the file into an array
	fileObj.close()
	return words

def findCommons(list1, list2):
	geness=[]
	for gene in list1:
		if (gene in list2):
			#print (gene)
			geness.append(gene)
	return geness

def retrieveARgs():
	inp=''
	emt=''
	try:
		opts, args = getopt.getopt(sys.argv[1:],"i:l:",["inputf=","list="]) #-i input -l list of emt genes
	except :
		print ('test.py -i <input folder> -l <EMTgenes.txt>')
		exit(2)
	for opt, arg in opts:
		if opt in ("-i", "--inputf"):
			inp = arg
		elif opt in ("-l", "--list"):
			emt = arg
	return inp, emt

###################################################################################################################
inputfolder, emtfile = retrieveARgs()
print ('Input file is', inputfolder)
print ('Output file is', emtfile)


files_dfs = [] 
for f in glob.glob(inputfolder+"/*"):
	
	if f.endswith('.csv') == False : 
		continue

	print (f)
	df = pd.read_csv(f,  sep=',', skiprows = 1, nrows=250)

	source=retrieveName(f)
	df['source_file']=source
	df.drop(["Fabs", "entrezgene_id", "hgnc_id", "uniprot_id","type", "ID"], axis=1, inplace=True)
	df.sort_values("rank", inplace=True)
	
	#This script merges on gene_name and not on every isoform to reduce complexity. The user must take this into consideration
	#and when proceeding when the analysis, better to take into account each isoform of the gene of interest. 
	df.drop_duplicates(subset=['gene_name'], keep='first',  inplace=True)
	files_dfs.append(df)
	#print (df)
	
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['gene_name'],
													how='outer'), files_dfs)
print(df_merged)
df_merged.to_csv('CTNNB1_significant.csv', index=False)  


genes = df_merged["gene_name"].to_numpy()
print (type(genes))

emt_genes = readFile(emtfile)
#print(emt_genes)

#emt_genes = np.array(emt)
print (type(emt_genes), len(emt_genes))

total = findCommons(genes, emt_genes)
print(total, len(total))