import argparse
from typing import Text
import os, sys
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Tool for scaling images to the required input size.")
    parser.add_argument(
        "--directory",
        type=Text,
        required=True,
        help="Path to folder containing images to scale."
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs='+',
        required=True,
        help="Size to scale the images to (height, width)."
    )
    parser.add_argument(
        "--mode",
        type=Text,
        required=True,
        help="--mode [scale, crop]"
    )
    args = parser.parse_args()

    if args.mode == "scale":
        scale_images(args)
    elif args.mode == "crop":
        crop_images(args)
    else:
        print("ERROR: Unknown mode: {}".format(args.mode))
        return 1


def scale_images(args):

    size = args.size
    if len(size) == 1:
        height = width = size[0]
    elif len(size) == 2:
        height = size[0]
        width = size[1]
    else:
        print("Dimensions to be scaled to is unknown: {}".format(size))
        return 1

    print("Scaling images to {} x {} (height x width)...".format(height, width))

    for file in os.listdir(args.directory):
        if file.endswith(".JPEG") and file.startswith("ILSVRC"):
            orig_img = Image.open(os.path.join(args.directory, file), "r")
            new_img = orig_img.resize((width, height), Image.ANTIALIAS)
            new_img.save(os.path.join(args.directory, "{}x{}_scale_".format(height, width) + file))
        else:
            continue

    return 0


def crop_images(args):

    size = args.size
    if len(size) == 1:
        height = width = size[0]
    elif len(size) == 2:
        height = size[0]
        width = size[1]
    else:
        print("Dimensions to be cropped to is unknown: {}".format(size))
        return 1

    print("Cropping images to {} x {} (height x width)...".format(height, width))

    for file in os.listdir(args.directory):
        if file.endswith(".JPEG") and file.startswith("ILSVRC"):
            orig_img = Image.open(os.path.join(args.directory, file), "r")

            x1 = orig_img.height/2 - height/2
            x2 = orig_img.height/2 + height/2
            y1 = orig_img.width/2 - width/2
            y2 = orig_img.width / 2 + width / 2

            print(x1, x2, y1, y2)

            new_img = orig_img.crop((x1, x2, y1, y2))
            new_img.save(os.path.join(args.directory, "{}x{}_crop_".format(height, width) + file))
        else:
            continue

    return 0


if __name__ == '__main__':
    main()