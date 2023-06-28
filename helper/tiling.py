# python tiling.py /path/to/yolo-format/folder /path/to/output/folder
# eg. python tiling.py /home/wenyi/DATA/synthetics/DOTA_yolo_small_vehs_only/ /home/wenyi/DATA/synthetics/DOTA_yolo_small_vehs_only_TILED/

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from shapely.geometry import Polygon

# Takes a Path object as input and creates a directory at the specified path if it doesn't already exist.
# The function handles potential errors and prints appropriate error messages in case of failure.
def create_directory(path):
  try:
    path.mkdir(parents=True, exist_ok=True)
  except FileNotFoundError:
    print(f"Specified directory {path} not found. Unable to create folder.")
  except PermissionError:
    print(f"Permission denied. Unable to create folder {path}.")
      
def create_tiles(img_filename, img, img_name, img_ext, op_imgs_path, op_labels_path, slice_size, boxes, negative_samples_path):
  imr = np.array(img, dtype=np.uint8)
  height, width = img.height, img.width

  # Create tiles and find intersection with bounding boxes for each tile
  for i in range((height // slice_size)):
    for j in range((width // slice_size)):
      x1 = j*slice_size
      y1 = height - (i*slice_size)
      x2 = ((j+1)*slice_size) - 1
      y2 = (height - (i+1)*slice_size) + 1

      pol = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
      no_annotations = True
      slice_labels = []

      for box in boxes:
        if pol.intersects(box[1]):
          inter = pol.intersection(box[1])    
          
          if no_annotations:
            sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
            sliced_im = Image.fromarray(sliced)
            rgb_im = sliced_im.convert('RGB')
            slice_path = str(op_imgs_path / f'{img_name}_{i}_{j}{img_ext}')
            
            slice_labels_path = str(op_labels_path / f'{img_name}_{i}_{j}.txt')
            
            rgb_im.save(slice_path)
            no_annotations = False          
          
          # Get the smallest polygon (with sides parallel to the coordinate axes) that contains the intersection
          new_box = inter.envelope 
          
          # Get central point for the new bounding box 
          centre = new_box.centroid
          
          # Get coordinates of polygon vertices
          try:
            x, y = new_box.exterior.coords.xy
          except AttributeError:
            print(f"AttributeError in: {img_filename}")
            continue
          
          # Get bounding box width and height normalized to slice size
          new_width = (max(x) - min(x)) / slice_size
          new_height = (max(y) - min(y)) / slice_size
          
          # We have to normalize central x and invert y for yolo format
          new_x = (centre.coords.xy[0][0] - x1) / slice_size
          new_y = (y1 - centre.coords.xy[1][0]) / slice_size

          slice_labels.append([box[0], new_x, new_y, new_width, new_height])
      
      # Save txt with labels for the current tile
      if len(slice_labels) > 0:
        slice_df = pd.DataFrame(slice_labels, columns=['class', 'x1', 'y1', 'w', 'h'])
        slice_df.to_csv(slice_labels_path, sep=' ', index=False, header=False, float_format='%.6f')
      
      # If negative_samples_path is indicated & there are no bounding boxes intersecting the current tile, save this tile to a separate folder
      if negative_samples_path and no_annotations:
        sliced = imr[i*slice_size:(i+1)*slice_size, j*slice_size:(j+1)*slice_size]
        sliced_im = Image.fromarray(sliced)
        rgb_im = sliced_im.convert('RGB')

        # Save the image file and generate the label file
        slice_path = Path(negative_samples_path) / 'images' / f'{img_name}_{i}_{j}{img_ext}'
        rgb_im.save(str(slice_path))
        empty_label_path = Path(negative_samples_path) / 'labels' / f'{img_name}_{i}_{j}.txt'
        with open(str(empty_label_path), 'w') as f:
          pass

        no_annotations = False

def tile_images(dir, op_dir, negative_samples_path, slice_size):
  # Create necessary folders
  img_path = dir / "images"
  labels_path = dir / "labels"
  op_imgs_path = op_dir / "images"
  op_labels_path = op_dir / "labels"
  create_directory(op_imgs_path)
  create_directory(op_labels_path)
  
  if negative_samples_path:
    create_directory(negative_samples_path / "images")
    create_directory(negative_samples_path / "labels")

  # Tile all images in a loop
  total_imgs = len(list(img_path.iterdir()))
  for t, img_filename in enumerate(img_path.iterdir()):
    if t % 100 == 0:
      print(f"=== {t} out of {total_imgs}")

    img_name, img_ext = img_filename.stem, img_filename.suffix
    try:
      img = Image.open(img_filename)
    except Image.DecompressionBombError:
      print(f"DecompressionBombError. Skipping image...")
      continue

    labels = pd.read_csv(labels_path / f'{img_name}.txt', sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
    
    # # we need to rescale coordinates from 0-1 to real image height and width
    labels[['x1', 'w']] = labels[['x1', 'w']] * img.width
    labels[['y1', 'h']] = labels[['y1', 'h']] * img.height
    
    boxes = []
    # Convert bounding boxes to shapely polygons. We need to invert Y and find polygon vertices from center points
    for row in labels.iterrows():
      x1 = row[1]['x1'] - row[1]['w'] / 2
      y1 = (img.height - row[1]['y1']) - row[1]['h'] / 2
      x2 = row[1]['x1'] + row[1]['w'] / 2
      y2 = (img.height - row[1]['y1']) + row[1]['h'] / 2

      boxes.append((int(row[1]['class']), Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])))
    
    create_tiles(img_filename, img, img_name, img_ext, op_imgs_path, op_labels_path, slice_size, boxes, negative_samples_path)

def main(dataset_directory, output_folder, negative_samples_path, size):
  # Printing config
  print(f"Slice size: {size}")

  output_path = Path(output_folder)

  # Folder Structure 1:
    # dataset_directory
    # ├── images
    # └── labels
  if (Path(dataset_directory) / "images").is_dir() and (Path(dataset_directory) / "labels").is_dir():
    # Start tiling
    print(f"Tiling '{dataset_directory}' into '{output_path}'")
    tile_images(Path(dataset_directory), output_path, negative_samples_path, size)

  # Folder Structure 2 (subfolders):
    # dataset_directory
    # ├── folder_1
    # │   └── images
    # │   └── labels
    # ├── folder_2
    # │   └── images
    # │   └── labels
    # └── ...
  else:
    sub_folders = [x for x in Path(dataset_directory).iterdir() if x.is_dir()]
    for i, sub_folder in enumerate(sub_folders):
      img_subfolder = (sub_folder / "images")
      label_subfolder = (sub_folder / "labels")

      if img_subfolder.is_dir() and label_subfolder.is_dir():
        op_folder = output_path / sub_folder.name

        # Start tiling for subfolder
        print(f"{i + 1} of {len(sub_folders)}: Tiling '{sub_folder}' into '{op_folder}'")
        tile_images(sub_folder, op_folder, negative_samples_path, size)

      else:
        print(f"Cannot find 'images' or 'labels' folder in {sub_folder}. Skipping tiling this folder...")
  
  print("Completed")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dataset_directory', type=Path, help='path to the dataset directory')
  parser.add_argument('output_folder', type=Path, help='path to the target output folder')
  parser.add_argument('--negative-samples-path', type=Path, help='path to where the negative samples should be saved')
  parser.add_argument('--size', type=int, default=640, help='slice size (default: 640)')
  args = parser.parse_args()

  main(args.dataset_directory, args.output_folder, args.negative_samples_path, args.size)
