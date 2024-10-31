import os
import csv
import tensorflow as tf
from utils.common import *
from model import SRCNN 
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int, default=4,     help='-')
parser.add_argument('--architecture', type=str, default="973135", help='-')
parser.add_argument("--ckpt-path",    type=str, default="",    help='-')

# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955", "18210", "18610", "181010", "1895210", "27315", "27915", "271515", "36420", "361220", "362020", "97315", "9315", "973135"]:
    raise ValueError("architecture must be 915, 935, 955, 18210, 18610, 181010, 1895210, 27315, 27915, 271515, 36420, 361220, 362020, 97315, 9315 or 973135")

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"

sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6
if architecture in ["915", "935", "955", "18210", "18610", "181010", "27315", "27915", "271515", "36420", "361220", "362020", "97315", "9315", "973135"]:
    pad = 0  # Adjust padding for architectures 18210, 18610, 181010, and 1895210 if needed

# -----------------------------------------------------------
# test
# -----------------------------------------------------------

def main():
    model = SRCNN(architecture)
    model.load_weights(ckpt_path)

    ls_data = sorted_list(f"dataset1/test/x{scale}/data")
    ls_labels = sorted_list(f"dataset1/test/x{scale}/labels")

    sum_psnr = 0
    list_psnr = []
    psnr = 0
    for i in range(0, len(ls_data)):
        lr_image = read_image(ls_data[i])
        lr_image = gaussian_blur(lr_image, sigma=sigma)
        bicubic_image = upscale(lr_image, scale)
        hr_image = read_image(ls_labels[i])
        if architecture not in ["915", "935", "955", "18210", "18610", "181010", "27315", "27915", "271515", "36420", "361220", "362020", "97315", "9315", "973135"]:
            hr_image = hr_image[pad:-pad, pad:-pad]

        bicubic_image = rgb2ycbcr(bicubic_image)
        hr_image = rgb2ycbcr(hr_image)

        bicubic_image = norm01(bicubic_image)
        hr_image = norm01(hr_image)

        bicubic_image = tf.expand_dims(bicubic_image, axis=0)
        sr_image = model.predict(bicubic_image)[0]

        # Ensure both images have the same shape
        min_shape = tf.minimum(tf.shape(hr_image), tf.shape(sr_image))
        hr_image = hr_image[:min_shape[0], :min_shape[1], :min_shape[2]]
        sr_image = sr_image[:min_shape[0], :min_shape[1], :min_shape[2]]

        psnr = PSNR(hr_image, sr_image, max_val=1).numpy()
        sum_psnr += psnr
        list_psnr.append((ls_data[i], psnr))

    avg_psnr = sum_psnr / len(ls_data)
    print(f"Average PSNR: {avg_psnr:.4f}")

    for filename, psnr_value in list_psnr:
        print(f"File: {os.path.basename(filename)} - PSNR: {psnr_value:.4f}")

    with open(f"{architecture}_{scale}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'PSNR'])
        for filename, psnr_value in list_psnr:
            writer.writerow([os.path.basename(filename), f"{psnr_value:.3f}"])

if __name__ == "__main__":
    main()
