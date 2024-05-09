import cv2 as cv
import numpy as np
import tools

from os.path import join

def fom(x, square_distance, i_y, a=1/9):
  i_x = np.sum(x) // 255
  i_n = max(i_x, i_y)
  return np.sum( (x / 255) / (1 + square_distance * a)) / i_n



def edge_detection(dir, modded_rgb, boundary_names, low_t=219, high_t=229):

  modded_boundary = []

  for (image, boundary) in zip(modded_rgb, boundary_names):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, low_t, high_t)
    blank = np.zeros((480,640), dtype='uint8')
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ext_contours = cv.drawContours(blank, contours, -1, (255, 255, 255), 1)
    cv.imwrite(join(dir, boundary), ext_contours)
    modded_boundary.append(ext_contours)
  
  print('Images writen to: ', dir)

  return modded_boundary

def edge_evaluation(dir, edges, names, square_distances, i_ys, save_name='_edge.csv'):

  results = [[None] * 2 for _ in range(len(square_distances)+1)]
  results[0][0] = 'mean figure of merit'

  for edge, square_distance, i_y, image, loc in zip(edges, square_distances, i_ys, names, range(len(names))):
    results[loc+1][0] = image
    results[loc+1][1] = fom(edge, square_distance, i_y, 1/9)
  
  # Mean
  results_np = np.array(results)
  results[0][1] = np.mean(results_np[1:, 1], axis=0)

  # Data export
  tools.csv_write(results, dir, save_name)

  return
