import numpy as np 
import os
import cv2
import random
import pickle

DIRECTORY = 'dataset'
CATEGORIES = ['mask_off', 'mask_on']
IMAGE_SIZE = 50
trainingData = []
featureSet = []
labelSet = []


def createTrainingData():

    for category in CATEGORIES:
    
        pathToCategories = os.path.join(DIRECTORY, category)
        categoryNumber = CATEGORIES.index(category)
        
        for image in os.listdir(pathToCategories):
            imageArray = cv2.imread(os.path.join(pathToCategories, image), \
                                    cv2.IMREAD_GRAYSCALE)
            resizedImageArray = cv2.resize(imageArray, (IMAGE_SIZE, IMAGE_SIZE))
            trainingData.append([resizedImageArray, categoryNumber])
        
        random.shuffle(trainingData)

def createFeatureAndLabelSets(featureSet, labelSet):
    
    for features, label in trainingData:
    
        featureSet.append(features)
        labelSet.append(label)

    featureSet = np.array(featureSet).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        

createTrainingData()
createFeatureAndLabelSets(featureSet, labelSet)

pickleOut = open('featureSet.pickle', 'wb')
pickle.dump(featureSet, pickleOut)
pickleOut.close()

pickleOut = open('labelSet.pickle', 'wb')
pickle.dump(labelSet, pickleOut)
pickleOut.close()
