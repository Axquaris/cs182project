import os, pathlib, glob
from imagenet_c import corrupt
from PIL import Image
from numpy import asarray


""" Instead of downlading the corrupted images, we simply apply the corruptions
    ourselves via the imagenet_e library. There are 15 corruption types with
    5 levels of severity making for 75 corrupted images per source image. """

source = "tiny-imagenet-200"
source_dir = pathlib.Path('./tiny-imagenet-200')
corrupt_dir = pathlib.Path('./tiny-imagenet-c')

if not os.path.exists(corrupt_dir):
    os.mkdir(corrupt_dir)

sub_dirs = ["train", "test", "val"]
corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', \
               'defocus_blur', 'motion_blur', \
               'zoom_blur', 'brightness', \
               'contrast', 'elastic_transform', 'pixelate', \
               'jpeg_compression', 'speckle_noise', 'gaussian_blur', \
               'spatter', 'saturate']

for sub_dir in sub_dirs:
    source_img_dir = os.path.join(source_dir, sub_dir)
    corrupt_img_dir = os.path.join(corrupt_dir, sub_dir)
    print(f"Source dir: {source_img_dir}")
    all_files = glob.glob(source_img_dir + "**/**/*.JPEG", recursive=True)
    num = 1
    for file in all_files:
        source_filename = os.fsdecode(file)
        img = asarray(Image.open(source_filename))

        for corruption in corruptions:
            for severity in range(6):
                suffix = os.path.relpath(source_filename,
                                         os.path.join(source_dir, sub_dir))
                dest_filename = f"{suffix}_{corruption}_{severity}"
                dest_filepath = os.path.join(corrupt_img_dir, dest_filename)
                corrupt_img = Image.fromarray(
                    corrupt(img, severity, corruption))
                if not os.path.exists(os.path.dirname(dest_filepath)):
                    os.makedirs(os.path.dirname(dest_filepath))
                corrupt_img.save(f"{dest_filepath}.JPEG")
        print(f"{num}/{len(all_files)} done for sub directory {sub_dir}")
        num += 1
    print(f"Sub directory {sub_dir} complete!")
