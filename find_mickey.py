#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Import packages 
import numpy as np 
import argparse
import imutils
import shutil
from pathlib import Path
import glob
import cv2
import pandas as pd
import os 

#Resized image
def resized(image, scale):
    return imutils.resize(image, width = int(image.shape[0] * scale))

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--template', required = True, help = 'Path to template images')
ap.add_argument('-i', '--images', required = True, help = 'Path to images where template will be matched')
ap.add_argument('-e', '--edges', required = False, default = 'canny', help = 'Type of edge detection available is "canny", "sobel", or "boundary"')
ap.add_argument('-o', '--output_path', required = False, default = os.getcwd(), help = 'Path to output directory')
args = vars(ap.parse_args())

#Create output image dir
output_dir = args['output_path']
dir_ = output_dir + '\\' + args['edges'] + '_outputs'
#Delete dir if it exists
if os.path.exists(dir_):
    shutil.rmtree(dir_)
os.makedirs(dir_)

#Create output data dir
data_dir = dir_ + '\\' 'output_data'
#Delete dir if it exists
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.makedirs(data_dir)

# loop over the images
for image_ in glob.glob(args['images'] + "/*.jpg"):
    print(image_)

    #Filename
    image_name = Path(image_).name[:-4]
    #load image
    image = cv2.imread(image_)
    color_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Remove noise from image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morph = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#     morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    
    #Erode image
#     kernel = np.ones((5,5),np.uint8)
#     morph = cv2.erode(morph,kernel,iterations = 1)
    
    image = morph
    
    #Define edge detection
    #Canny edge detection
    if args['edges'].lower() == 'canny':
        lower_thresh = np.mean(image) * 0.66
        upper_thresh = np.mean(image) * 1.33
        image = cv2.Canny(image, lower_thresh, upper_thresh)

    #Sobel edge detection
    if args['edges'].lower() == 'sobel':
        #Filter image 
        sobelx_5x5 = cv2.Sobel(image, -1, 1, 0, 5)
        sobely_5x5 = cv2.Sobel(image, -1, 0, 1, 5)
        image = sobelx_5x5 + sobely_5x5

    #Boudary edges
    if args['edges'].lower() == 'boundary':
        #Convert to binary 
        thresh = np.mean(image) * 0.66
        #Binary image
        binary_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
        #Generate structured kernel 
        struct_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        #Erode binary image
        eroded_img = cv2.erode(binary_img, struct_kernel, iterations = 1)
        #I - B(I)
        image = binary_img - eroded_img
        
    #Processing loop scaling the template
    scores = {}
    #Data of structure (template dims, locs, angle, template)
    data = {}
    #Set file index
    n = 0

    #Loop through templates 
    for template_ in glob.glob(args['template'] + "/*.jpg"):
        print(template_)
        #Template name
        template_name = Path(template_).name[:-4]
        #Load template and find edges
        template = cv2.imread(template_)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        #Template edges
        template = cv2.Canny(cv2.bitwise_not(template), 80, 200)
               
        #Loop through different template scales
        for scale in np.linspace(0.3, 1.5, 20):
            #Loop through different template rotations
            for angle in np.linspace(0, 360, 73)[:-1]:
                print(template_name + '_' + str(scale) + '_' + str(angle))
                #Rotate template
                template_rotated = imutils.rotate(template, angle)
                #Resize template
                resized_template = resized(template_rotated, scale)
                #Dims of resized template
                template_h = resized_template.shape[1]
                template_w = resized_template.shape[0]
                #Image shape
                image_h = image.shape[1]
                image_w = image.shape[0]
                #Break from loop if the template becomes bigger than the image
                if template_h > image_h or template_w > image_w:
                    break  
                #Run template through image
                result = cv2.matchTemplate(image, resized_template, cv2.TM_CCORR_NORMED)
                #Get matching score and location
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
                #Add correlation value and result to scores
                scores[str(maxVal)] = result
                #Add info to data
                data[str(maxVal)] = ((template_h, template_w), (maxLoc[0], maxLoc[1]), angle, template_name)
#                                      , args['edges'])
#     try: 
    #Find best match from scores
    best_match = max(scores.keys())
    print(data[best_match])
    #Output color image
    image_copy = color_image.copy()
    #Create rectangle around best match
    cv2.rectangle(image_copy, (data[best_match][1][0], data[best_match][1][1]), (data[best_match][1][0] + data[best_match][0][1], data[best_match][1][1] + data[best_match][0][0]), (255, 255, 255), 2)
    #Write image
    cv2.imwrite(dir_ + '\\' + image_name + template_name + data[best_match][3] + str(data[best_match][2]) + '_annotated.jpg', image_copy)

    #Write data
    df = pd.DataFrame([data[best_match]])
    df.to_csv(data_dir + '\\' + image_name + '_data_output.csv')
    n += 1    
#     except Exception: 
#         pass