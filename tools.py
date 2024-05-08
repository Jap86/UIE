from typing import Tuple
from typing import List
import random
from os.path import exists, join
import csv

def im_exists(dir1: str, batch_size: int = 10) -> Tuple[List[str],List[str]]:

  # Ensure given directories exist and are coherent
  assert exists(dir1) == True, 'Given image directory '+dir1+' must exist'

  # Full and empty name lists
  full_rgb_filenames = ['{:05d}.png'.format(i) for i in range(1, 10256)]

  filename_exist = []

  # Add existing filenames to list
  for filename in full_rgb_filenames:
    if exists(join('dataset/rgb', filename)):
      filename_exist.append(filename)

  rgb_names = random.sample(filename_exist, batch_size)
  boundary_names = [image.split('.')[0] + '_edge.png' for image in rgb_names]

  return rgb_names, boundary_names

def csv_write(data: List[str], filepath: str, filename: str):
  # Write data 'data', to csv file 'filename' at location 'filepath'
  directory = join(filepath, filename)
  with open(directory, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(data)
    print('Data successfully writen to: ', directory)
  return