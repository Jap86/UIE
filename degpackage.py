import cv2 as cv
from os.path import join
import numpy as np

# Random noise
def add_random_noise(deg, rgb, filenames, stddev=0.5):
  for image, file in zip(rgb, filenames):
    noise = np.random.normal(loc=0, scale=stddev, size=image.shape).astype(np.uint8)
    noisy_image = cv.add(image, noise)

    cv.imwrite(join(deg, 'enh_none', file), noisy_image)
  
  return noisy_image

# Haze
def add_haze(deg, rgb, filenames, haze_colour=(122, 150, 0), haze_intensity=0.25):
  for image, file in zip(rgb, filenames):
    haze = np.ones_like(image, dtype=np.uint8)
    haze[:, :] = haze_colour
        
    rgb_haze = cv.cvtColor(haze, cv.COLOR_BGR2RGB)
    hazy_image = cv.addWeighted(image, 1 - haze_intensity, rgb_haze, haze_intensity, 0)

    cv.imwrite(join(deg, 'enh_none', file), hazy_image)

  return

# Depth-aware haze
def add_haze_depth(deg, rgb, depth_images, filenames, haze_colour=(122, 150, 0), haze_intensity=0.25, depth_intensity=0.5):

  for image, depth, file in zip(rgb, depth_images, filenames):
    haze_depth_image = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)

    normalised_depth_intensity = (1 - ((haze_depth_image / 255.0) * depth_intensity))
    haze = (haze_colour * normalised_depth_intensity[..., np.newaxis]).astype(np.uint8)

    rgb_haze = cv.cvtColor(haze, cv.COLOR_BGR2RGB)
    hazy_image = cv.addWeighted(image, 1 - haze_intensity, rgb_haze, haze_intensity, 0)

    cv.imwrite(join(deg, 'enh_none', file), hazy_image)

  return

# Reduce contrast and brightness
def reduce_contr_bright(deg, rgb, filenames, contrast_factor: float = 0.8, brightness_factor: float = -5):
  # contrast_factor: 0 < cf < 1 = reduced contrast
  # brightness factor: bf < 0 = reduced brightness
  for image, file in zip(rgb, filenames):
    reduced_image = cv.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
    cv.imwrite(join(deg, 'enh_none', file), reduced_image)
  return

def add_none(deg, rgb, filenames):
  for image, file in zip(rgb, filenames):
    cv.imwrite(join(deg, 'enh_none', file), image)
  return

def degrade(deg, rgb, depth_images, filenames):
  

  if deg == 'deg_random_noise':
    add_random_noise(deg, rgb, filenames)
  if deg == 'deg_haze':
    add_haze(deg, rgb, filenames)
  if deg == 'deg_haze_depth':
    add_haze_depth(deg, rgb, depth_images, filenames)
  if deg == 'deg_reduce_contr_bright':
    reduce_contr_bright(deg, rgb, filenames)
  if deg == 'deg_none':
    add_none(deg, rgb, filenames)

  return