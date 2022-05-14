# This is a sample Python script.
import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd='C:\\Users\\bkm\\Downloads\\tesseract.exe'

roi=[(35,765),(1450,930)]

img1=cv2.imread('form.jpg')

h,w,c=img1.shape

orb=cv2.ORB_create(6000)
kp1,des1 = orb.detectAndCompute(img1,None)
#imgKp1=cv2.drawKeypoints(imgq,kp1,None)
bf=cv2.BFMatcher(cv2.NORM_HAMMING)

img2=img1.copy()
imgmask=np.zeros_like(img2)
cv2.rectangle(imgmask,(roi[0]),(roi[1]),(0,255,0),cv2.FILLED)
img2=cv2.addWeighted(img2,0.99,imgmask,0.1,0)


imgCrop=img2[33:765,450:130]
cv2.imshow('crop',imgCrop)

cv2.waitKey(0)
#cv2.imshow('keypoint',imgKp1)
cv2.imshow('output',img2)
cv2.waitKey(0)