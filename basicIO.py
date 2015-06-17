# -*- coding: utf-8 -*-
"""
Basic I/O:
         
     Aims at providing easy access to basic I/O operations with Python.
     
     [Volumns]:
     
     1. Deal with folders
         1.1 Create folder
         1.2 Move folder
     2. Deal with files
         2.1 Read/Write txt files
         2.2 Read/Write Json files
         2.3 Copy/Move Generic Files
         2.4 Execute JAR files
     3. Deal with data
         3.1 Read/Write Labels(txt format)
         3.2 Read/Write Features(numpy/pickle/h5 format)
         3.3 Possible imports needed for Machine Learning tasks

Created on Tue Jun 16 17:23:22 2015

@author: Guanghan Ning
"""


"""-------------------------------------------------------------------------"""
"""1. Deal with folders"""
      #1.1 Create folder
      #1.2 Move folder


"""1.1 Create folder"""
#START
import os
output_folder= "G:\\1.Data\\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
#END


"""1.2 Move folder"""
#START
import shutil
shutil.move
#END


"""-------------------------------------------------------------------------"""
"""2. Deal with files"""
         #2.1 Read/Write txt files
         #2.2 Read/Write Json files
         #2.3 Copy/Move Generic Files
         #2.4 Execute JAR files


"""2.1 Read/Write txt files""" 
#START
import os

txtfile = open("G:\\1.Data\\SceneClassification-training\\output_txt\\new.txt", "w", encoding='utf-8')
folder= 'G:\\1.Data\\SceneClassification-training\\output\\'
subfolders= [x[0] for x in os.walk(folder)]

for subfolder in subfolders:
	print(subfolder)
	for file in os.listdir(subfolder):
	    if file.endswith(".jpg"):
             partial = os.path.basename(subfolder)             
             filePath= '/' + partial + '/' + file  
             
             txtfile.write(filePath + '\n')
txtfile.close()     
#END


"""2.2 Read/Write Json files""" 
#START
import shutil
import os
import json
  
def parseImageName(fullPath):
    if len(fullPath)-1< 0:
        return -1
    parts = fullPath.split('/')
    #print('imageName', parts[len(parts)- 1])
    return parts[len(parts)- 1]

def parseImageTag(tags):
    if None in tags:
        return -1
    match = [x for x in tags if "place" in x]
    if len(match)< 1:
        return -1
    parts = match[0].split(':')
    #print(parts[len(parts)-1])
    if len(parts)-1 < 0:
        return -1
    else:
        return parts[len(parts)-1]

def processJsonFile(filePath, subfolder):
      #load single file
	with open(filePath,encoding='utf-8') as f:
         data = f.read()
         parsed_json = json.loads(data)
         #parse single file for image name and place tag
         for lines in parsed_json:
             #derive the images with their class labels[scene]
             if lines['imgkey']== None:
                 continue
             imageName= parseImageName(lines['imgkey'])
             imageTag= parseImageTag(lines['tags']) 
             if imageTag== -1:
                 continue

             #we can also get images with specific tags only             
             """
             if imageTag!= '汽车类-轿车':
                 continue
             """
             #build a folder to store the output images
             tempFolder= 'G:\\1.Data\\SceneClassification-training\\output2\\' + imageTag
             if not os.path.exists(tempFolder):
                 os.makedirs(tempFolder)  
             #copy the images to the destination folder
             partial= os.path.basename(subfolder)
             src= subfolder + '\\' + imageName
             dst= tempFolder + '\\' + partial + '_' + imageName
             shutil.copy(src, dst)
         #output cheers
         print('json file process success!')

#process the images 
folder= 'G:\\1.Data\\SceneClassification-training\\input2\\'
subfolders= [x[0] for x in os.walk(folder)]
for subfolder in subfolders:
	print(subfolder)
	for file in os.listdir(subfolder):
	    if file.endswith(".json"):
             filePath= subfolder + '\\' + file             
             print(filePath)
             processJsonFile(filePath, subfolder)
#END             
                       
              

"""2.3 Copy/Move Generic Files"""
#START
src= 'G:\\1.Data\\SceneClassification-training\\_test2\\s3ImageLoader-0.1.jar'
dst= 'G:\\1.Data\\SceneClassification-training\\_test2\\s3ImageLoader-0.1_copy.jar'
shutil.copyfile(src, dst)
shutil.move(src, dst)
#END


"""2.4 Execute JAR files"""
#START
import subprocess
import os

jsonFileFolder = 'G:\\1.Data\\SceneClassification-training\\films\\' 

subfolders= [x[0] for x in os.walk(jsonFileFolder)]

src= 'G:\\1.Data\\SceneClassification-training\\_test2\\s3ImageLoader-0.1.jar'

for subfolder in subfolders:
    print(subfolder)
    dst= subfolder + '\\s3ImageLoader-0.1.jar'
    #We can choose to copy file or just move file
    """shutil.copyfile(src, dst)"""
    shutil.move(src, dst)
    p= subprocess.call(['java', '-jar', dst])
    src= dst
#END


"""-------------------------------------------------------------------------"""
"""3. Deal with data"""
         #3.1 Read/Write Labels(txt format)
         #3.2 Read/Write Features(numpy format)
         #3.3 Possible imports needed for Machine Learning tasks


"""3.1 Read/Write Labels(txt format)"""
#START
def load_labels(file_name):
    input_file = open(file_name)
    lines = input_file.readlines()
    n_samples = len(lines)
    labels = []
    for line in lines:
        data = line.split()
        #labels.append([int(data[1])])
        labels.append(map(int, data[1].split(",")))           
        n_classes = max(labels)
        n_samples = len(labels)
        
    return labels,n_classes,n_samples
#END


"""3.2 Read/Write Features(numpy/pickle/h5 format)"""
#START
def load_features(file_name):

    #numpy format
    if file_name.endswith('npy'):
        features = np.load(file_name)

    #pickle format
    elif file_name.endswith('feat'):
        with open(file_name, 'rb') as fid:
            features = cPickle.load(fid)

    #h5 format
    else:
        df = pd.read_hdf(file_name, 'df')
        features = np.vstack(df.prediction.values)

    n_samples = features.shape[0]
    n_features = features.shape[1]    
    
    return features,n_features,n_samples
#END


"""3.3 Possible imports needed for Machine Learning tasks"""
#START
import numpy as np
import sklearn as sk
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import cPickle 
#END








