import cv2 as cv
import numpy as np
from os.path import join
import tools

from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as sk_mse

def eval_quality(dataset_rgb, dir, filenames, save_name='_quality.csv'):

  # Results array
  results = [[None] * 7 for _ in range(len(filenames)+1)]
  results[0][0] = 'mean quality results'
  for row in results:
      row[1] = 'psnr'
      row[3] = 'rmse'
      row[5] = 'ssim'
  
  for (rgb, image, loc) in zip(dataset_rgb, filenames, range(len(filenames))):
    results[loc+1][0] = image
    modded_image = cv.imread(join(dir, image))

    # Peak Signal-to-Noise Ratio
    results[loc+1][2] = cv.PSNR(rgb, modded_image, 255)
    # Root Mean Square Error
    results[loc+1][4] = np.sqrt(sk_mse(rgb, modded_image))
    # Structural Similarity Index Metric
    results[loc+1][6] = sk_ssim(rgb, modded_image, data_range=modded_image.max() - modded_image.min(), multichannel=True, channel_axis=2) 

  # Mean
  results_np = np.array(results)
  results[0][2] = np.mean(results_np[1:, 2], axis=0)
  results[0][4] = np.mean(results_np[1:, 4], axis=0)
  results[0][6] = np.mean(results_np[1:, 6], axis=0)

  # Data export
  tools.csv_write(results, dir, save_name)

  return