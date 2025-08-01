import os
import json
from PIL import Image
from tqdm import tqdm

# Class map (your custom YOLO class list)
label_map = {
    'car': 0,
    'person': 1,
    'traffic sign': 2
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert [x1, y1, x2, y2] box to YOLO format"""
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height

def process_split(split):
    json_dir = f'../data/labelsJson/{split}'
    img_dir = f'../data/images/{split}'
    out_dir = f'../data/labels/{split}'

    os.makedirs(out_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(json_dir)):
        if not file_name.endswith('.json'):
            continue

        json_path = os.path.join(json_dir, file_name)
        image_name = file_name.replace('.json', '.jpg')
        image_path = os.path.join(img_dir, image_name)

        # Load image to get size
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Load annotation
        with open(json_path, 'r') as f:
            data = json.load(f)

        objects = data['frames'][0]['objects']
        yolo_lines = []

        for obj in objects:
            category = obj['category']
            if category not in label_map or 'box2d' not in obj:
                continue
            class_id = label_map[category]
            box2d = obj['box2d']
            bbox = [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']]
            yolo_bbox = convert_bbox_to_yolo(bbox, img_width, img_height)
            line = f"{class_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}"
            yolo_lines.append(line)

        # Save .txt file
        out_path = os.path.join(out_dir, file_name.replace('.json', '.txt'))
        with open(out_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Optional: print file summary
        # print(f"Converted {file_name} → {len(yolo_lines)} boxes")

# 🔁 Process all three splits
for split in ['train', 'val', 'test']:
    process_split(split)
