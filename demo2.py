import os
import argparse
import tensorflow as tf
from utils.common import *
from model import SRCNN

def process_image(image_path, architecture, ckpt_path, scale, pad, sigma):
    # Baca gambar
    lr_image = read_image(image_path)
    
    # Pratinjau gambar yang dibaca
    print(f"Original Image Shape: {lr_image.shape}")

    # Proses gambar untuk arsitektur tertentu
    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)

    # Konversi ke YCbCr dan normalisasi
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)

    # Prediksi dengan model
    bicubic_image = tf.expand_dims(bicubic_image, axis=0)
    model = SRCNN(architecture)
    model.load_weights(ckpt_path)
    sr_image = model.predict(bicubic_image)[0]

    # Resize SR image to match HR image's size minus padding
    sr_image = sr_image[pad:-pad, pad:-pad]
    
    # Jika gambar dalam grayscale (1 channel), ubah ke RGB (3 channels)
    if sr_image.shape[-1] == 1:
        sr_image = tf.image.grayscale_to_rgb(sr_image)
    
    # Konversi kembali ke bentuk RGB
    sr_image = ycbcr2rgb(sr_image)
    sr_image = denorm01(sr_image)

    # Simpan hasil
    save_path = os.path.splitext(image_path)[0] + f'_SR_{architecture}.png'
    save_image(sr_image, save_path)
    print(f"Saved SR image to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help='Scaling factor')
    parser.add_argument('--architecture', type=str, default="9315", help='SRCNN architecture')
    parser.add_argument('--ckpt-path', type=str, default="", help='Path to the checkpoint')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the input image')

    FLAGS = parser.parse_args()

    # Validasi arsitektur dan skala
    architecture = FLAGS.architecture
    if architecture not in ["915", "935", "955", "18210", "18610", "181010", "1895210", "27315", "27915", "271515", "97315", "9315"]:
        raise ValueError("Invalid architecture")
    
    scale = FLAGS.scale
    if scale not in [2, 3, 4]:
        raise ValueError("Invalid scale factor")

    ckpt_path = FLAGS.ckpt_path if FLAGS.ckpt_path else f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"
    image_path = FLAGS.image_path

    sigma = 0.3 if scale == 2 else 0.2
    pad = int(architecture[1]) // 2 + 6

    process_image(image_path, architecture, ckpt_path, scale, pad, sigma)

if __name__ == "__main__":
    main()
