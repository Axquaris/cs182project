import os, re, shutil

curr_dir = os.path.dirname(os.path.abspath(__file__))

default_root_dir = os.path.join(curr_dir, "tiny-imagenet-200", "val")
default_img_dir = os.path.join(default_root_dir, "images")
default_annotation_file = os.path.join(default_root_dir, "val_annotations.txt")

def parse_annotations(annotations_file):
    label_to_img = {}
    with open(annotations_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        img, label = line.split()[0], line.split()[1]
        assert re.search("^val_\d+", img) and re.match("n\d{8}$", label), "label or image not formatted properly"
        if not label in label_to_img:
            label_to_img[label] = []
        label_to_img[label].append(img)
    
    return label_to_img

def main(root_dir=default_root_dir, img_dir=default_img_dir, annotations_file=default_annotation_file):
    label_to_img = parse_annotations(annotations_file)

    for label in label_to_img:
        label_dir = os.path.join(root_dir, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        for img in label_to_img[label]:
            prev_path = os.path.join(img_dir, img)
            new_path = os.path.join(label_dir, img)
            shutil.move(prev_path, new_path) 

if __name__ == '__main__':
    main()