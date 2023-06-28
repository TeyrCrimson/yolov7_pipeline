# This program annotates images in the YOLO format. It takes a dataset folder or a single image as input and displays each image for annotation. 
# Press 'q' to exit or any other key to move to the next image.

# Usage: python annotate_images.py /path/to/dataset/folder -s .jpg -n image1
from pathlib import Path
import random
import cv2
import argparse
from glob import glob

def generate_random_rgb():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return (r, g, b)

def annotate_images(folder, files_to_annotate, img_suffix):
  for name in files_to_annotate:
    print(f"Image: {name}")
    label_path = folder / "labels" / (name + ".txt")
    img_path = folder / "images" / (name + img_suffix)

    with open(label_path) as f:
      label_data = f.readlines()

    image = cv2.imread(str(img_path))
    img_height, img_width, _ = image.shape

    categories = {}
    for label in label_data:
      category, xc, yc, w, h = label.split()
      if category in categories:
        color = categories[category]
      else:
        color = generate_random_rgb()
        categories[category] = color

      width = float(w) * img_width
      height = float(h) * img_height
      x = float(xc) * img_width - width / 2.0
      y = float(yc) * img_height - height / 2.0
      cv2.rectangle(image, (int(x), int(y)), (int(x + width), int(y + height)), color, 2)
    
    cv2.imshow(name, image)
    key = cv2.waitKey(0) & 0xFF  # Capture the key pressed

    if key == ord('q'):  # If 'q' is pressed, break out of the loop
      break

    cv2.destroyAllWindows()

def main():
  parser = argparse.ArgumentParser(description="Annotate an image with the YOLO format")
  parser.add_argument("folder", type=Path, help="Path to the dataset folder (i.e. contains the 'images' and 'labels' folders)")
  parser.add_argument("-s", "--suffix", default=".png", help="Image suffix (default: .png)")
  parser.add_argument("-n", "--name", help="Name of specific image without the ext (optional)")

  args = parser.parse_args()
  print(args)

  files_to_annotate = []
  if args.name:
    files_to_annotate.append(args.name)
  else:
    for img in glob(str(args.folder / "labels" / "*.txt")):
      filename = Path(img).stem
      files_to_annotate.append(filename)

  annotate_images(args.folder, files_to_annotate, args.suffix)

if __name__ == "__main__":
    main()
