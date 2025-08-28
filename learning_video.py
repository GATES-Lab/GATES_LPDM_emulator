from PIL import Image
import os

def make_gif_with_pillow(input_dir, output_path="output.gif", duration=200):
    """
    Create a GIF from PNGs using Pillow.
    duration is in milliseconds per frame.
    """
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])
    if not files:
        raise ValueError("No PNG files found in the directory.")

    frames = [Image.open(os.path.join(input_dir, f)) for f in files]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF saved to {output_path}")

train_img_path = 'training_output/143763_SAHARA_200GB_2014_16_test_2016_13_timedelta_6_12_24/training_imgs'

make_gif_with_pillow(train_img_path, "animation.gif", duration=300)