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


def enhance(enh, deg, depth, filenames):
  
  rgb = []

  for file in filenames:
    rgb.append(cv.imread(join(deg, 'enh_none', file)))

  if enh == 'enh_increase_contr_bright':
    increase_contr_bright(deg, rgb, filenames)

  return