import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import SRCNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int, default=2, help='-')
parser.add_argument('--architecture', type=str, default="915", help='-')
parser.add_argument("--ckpt-path",    type=str, default="", help='-')
parser.add_argument("--image-path",   type=str, default="dataset1/test/x2/data/ob1.png", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path

architecture = FLAGS.architecture
if architecture not in ["915", "935", "955", "18210", "18610", "181010", "27315", "27915", "271515", "36420", "361220", "362020", "97315", "9315", "973135"]:
    raise ValueError("architecture must be 915, 935, 955, 18210, 18610, 181010, 27315, 27915, 271515, 36420, 361220, 362020, 97315, 9315 or 973135")

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("scale must be 2, 3, or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"

sigma = 0.3 if scale == 2 else 0.2
pad = int(architecture[1]) // 2 + 6


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = bicubic_image[pad:-pad, pad:-pad]
    write_image("bicubic1.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)
    bicubic_image = tf.expand_dims(bicubic_image, axis=0)

    model = SRCNN(architecture)
    model.load_weights(ckpt_path)
    sr_image = model.predict(bicubic_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)

    # Tambahkan dimensi tambahan untuk memastikan sr_image adalah grayscale dengan dimensi terakhir ukuran 1
    sr_image = tf.expand_dims(sr_image, axis=-1)

    # Sekarang konversi sr_image dari grayscale ke RGB
    sr_image = tf.image.grayscale_to_rgb(sr_image)

    sr_image = ycbcr2rgb(sr_image)
    
    # Hapus dimensi batch sebelum menyimpan gambar
    sr_image = tf.squeeze(sr_image, axis=0)
    
    write_image("ob1.png", sr_image)

if __name__ == "__main__":
    main()