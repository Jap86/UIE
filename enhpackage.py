import cv2 as cv
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

def increase_contr_bright(deg, rgb, filenames, contrast_factor: float = 1.3, brightness_factor: float = 15):
  # contrast_factor: 0 < cf < 1 = reduced contrast
  # brightness factor: bf < 0 = reduced brightness
  for image, file in zip(rgb, filenames):
    enh_image = cv.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
    cv.imwrite(join(deg, 'enh_increase_contr_bright', file), enh_image)
  return

def apply_hist_eq(deg, rgb, filenames):

  for image, file in zip(rgb, filenames):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    enh_image = cv.equalizeHist(gray)
    cv.imwrite(join(deg, 'enh_hist_eq', file), enh_image)

  return

def apply_clahe(deg, rgb, filenames, clip_limit=2.0, gridsize=(8,8)):

  for image, file in zip(rgb, filenames):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=gridsize)
    enh_image = clahe.apply(gray)
    cv.imwrite(join(deg, 'enh_clahe', file), enh_image)

  return

def enhance(enh, deg, filenames):
  
  rgb = []

  for file in filenames:
    rgb.append(cv.imread(join(deg, 'enh_none', file)))

  if enh == 'enh_increase_contr_bright':
    increase_contr_bright(deg, rgb, filenames)

  if enh == 'enh_hist_eq':
    apply_hist_eq(deg, rgb, filenames)

  if enh == 'enh_clahe':
    apply_clahe(deg, rgb, filenames)


  return