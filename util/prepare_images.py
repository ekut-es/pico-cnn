import argparse
from typing import Text
import os
from PIL import Image
import math

__author__ = "Alexander Jung (University of Tuebingen, Chair for Embedded Systems)"


def main():
    parser = argparse.ArgumentParser(description="Tool for preparing images (scaling/cropping) to the required input size.")
    parser.add_argument(
        "--directory",
        type=Text,
        required=True,
        help="Path to folder containing the images."
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs='+',
        required=True,
        help="Size to scale the images to (width, height)."
    )
    parser.add_argument(
        "--mode",
        type=Text,
        required=True,
        help="--mode [scale, crop, center_crop]"
    )
    parser.add_argument(
        "--crop_pos",
        type=int,
        nargs='+',
        required=False,
        help="Position of top left corner of non-center crop (x, y). Note that this position has to satisfy  \
              the following conditions: 0 < x+width < orig_width and 0 < y+height < orig_height."
    )
    args = parser.parse_args()

    if args.mode == "scale":
        scale_images(args)
    elif args.mode == "crop":
        crop_images(args)
    elif args.mode == "center_crop":
        center_crop_images(args)
    else:
        print("ERROR: Unknown mode: {}".format(args.mode))
        return 1


def scale_images(args):

    size = args.size
    if len(size) == 1:
        width = height = size[0]
    elif len(size) == 2:
        width = size[0]
        height = size[1]
    else:
        print("Dimensions to be scaled to is unknown: {}".format(size))
        return 1

    destination_folder = args.directory + "/scaled_{}x{}".format(width, height)

    try:
        os.makedirs(destination_folder)
    except FileExistsError:
        pass

    print("Scaling images to {} x {} (width x height)...".format(width, height))

    files = os.listdir(args.directory)

    for num, file in enumerate(files):
        print("Processing {} ({}/{})".format(file, num+1, len(files)))
        # if file.endswith(".JPEG") and file.startswith("ILSVRC"):
        if file.endswith(".JPEG"):
            orig_img = Image.open(os.path.join(args.directory, file), "r")
            new_img = orig_img.resize((width, height), Image.ANTIALIAS)
            new_img.save(os.path.join(destination_folder, "{}x{}_scale_".format(width, height) + file))
        else:
            print("Skipping non-image file.")
            continue

    return 0


def center_crop_images(args):

    size = args.size
    if len(size) == 1:
        width = height = size[0]
    elif len(size) == 2:
        width = size[0]
        height = size[1]
    else:
        print("Dimensions to be cropped to is unknown: {}".format(size))
        return 1

    destination_folder = args.directory + "/center_cropped_{}x{}".format(width, height)

    try:
        os.makedirs(destination_folder)
    except FileExistsError:
        pass

    print("Center cropping images to {}x{} (width x height)...".format(width, height))

    files = os.listdir(args.directory)

    for num, file in enumerate(files):
        print("Processing {} ({}/{})".format(file, num+1, len(files)))
        if file.endswith(".JPEG"):
            orig_img = Image.open(os.path.join(args.directory, file), "r")

            orig_width, orig_height = orig_img.size

            left = math.floor((orig_width - width)/2)
            top = math.floor((orig_height - height)/2)
            right = math.floor((orig_width + width)/2)
            bottom = math.floor((orig_height + height)/2)

            new_image = orig_img.crop((left, top, right, bottom))
            new_image.save(os.path.join(destination_folder, "{}x{}_center_crop_".format(width, height) + file))
        else:
            print("Skipping non-image file.")
            continue

    return 0


def crop_images(args):

    size = args.size
    if len(size) == 1:
        width = height = size[0]
    elif len(size) == 2:
        width = size[0]
        height = size[1]
    else:
        print("Dimensions to be cropped to is unknown: {}".format(size))
        return 1

    position = args.crop_pos

    if position is None:
        print("Please specify the top left corner of the desired crop (--crop_pos)")
        return 1

    if len(position) == 1:
        pos_x = pos_y = position[0]
    elif len(position) == 2:
        pos_x = position[0]
        pos_y = position[1]
    else:
        print("Position of top left crop corner is unknown: {}".format(position))
        return 1

    destination_folder = args.directory + "/cropped_{}x{}".format(width, height)

    try:
        os.makedirs(destination_folder)
    except FileExistsError:
        pass

    print("Cropping images to {}x{} (width x height)...".format(width, height))

    files = os.listdir(args.directory)

    for num, file in enumerate(files):
        print("Processing {} ({}/{})".format(file, num+1, len(files)))
        if file.endswith(".JPEG") and file.startswith("ILSVRC"):
            orig_img = Image.open(os.path.join(args.directory, file), "r")

            assert (pos_x + width) < orig_img.width
            assert (pos_y + height) < orig_img.height

            x1 = pos_x
            x2 = pos_x + width
            y1 = pos_y
            y2 = pos_y + height

            # print(x1, y1, x2, y2)

            new_img = orig_img.crop((x1, y1, x2, y2))
            new_img.save(os.path.join(destination_folder, "{}x{}_crop_".format(width, height) + file))
        else:
            print("Skipping non-image file.")
            continue

    return 0


if __name__ == '__main__':
    main()
