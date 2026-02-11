from pathlib import Path
import yaml
import json
import os
import random
import cv2
import numpy as np
import shutil

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

create_dataset_dir = project_root / "create_dataset"
raw_images_dir = create_dataset_dir / "raw_image"
raw_labels_dir = create_dataset_dir / "labels"
dataset_dir = project_root / "dataset"
tmp_dir = dataset_dir / "tmp"
data_yaml_path = dataset_dir / "data.yaml"
config_file = project_root / "config" / "config.json"

with open(config_file, "r") as f:
    config = json.load(f)
image_size = config["image_size"]
train_ratio = config["train_ratio"]
val_ratio = config["val_ratio"]
classes = config["classes"]

print(f"Cấu hình Dataset:")
print(f"  Kích thước ảnh: {image_size}")
print(f"  Tỷ lệ train: {train_ratio}")
print(f"  Tỷ lệ val: {val_ratio}")
print(f"  Tỷ lệ test: {1 - train_ratio - val_ratio}")

def normalize_extensions(folder: Path):
    """Chuyển tất cả extension ảnh về lowercase"""
    count = 0
    for ext in ["*.JPG", "*.JPEG", "*.PNG"]:
        for path in folder.glob(ext):
            new_path = path.with_suffix(path.suffix.lower())
            if path != new_path:
                path.rename(new_path)
                count += 1
    print(f"Đã chuẩn hóa {count} file extensions")

normalize_extensions(raw_images_dir)

def load_label(txt_path):
    # class x1 y1 x2 y2 x3 y3 x4 y4
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = parts[1:]
            boxes.append((cls, coords))
    return boxes

def save_label(path, labels):
    with open(path, "w") as f:
        for cls, coords in labels:
            f.write(f"{cls} " + " ".join(f"{x:.6f}" for x in coords) + "\n")

def resize_img_and_label(img, labels, size):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (size, size))

    new_labels = []
    for cls, coords in labels:
        new_labels.append((cls, coords))

    return img_resized, new_labels

def rotate_90(img, labels):
    """Xoay 90 độ theo chiều kim đồng hồ"""
    img_r = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Rotate 90: (x, y) -> (1-y, x)
            nx = 1 - y
            ny = x
            new_c += [nx, ny]
        new.append((cls, new_c))
    return img_r, new

def rotate_180(img, labels):
    """Xoay 180 độ"""
    img_r = cv2.rotate(img, cv2.ROTATE_180)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            new_c += [1-x, 1-y]
        new.append((cls, new_c))
    return img_r, new

def rotate_270(img, labels):
    img_r = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    new = []
    for cls, coords in labels:
        new_c = []
        for i in range(4):
            x = coords[i*2]
            y = coords[i*2+1]
            # Rotate 270: (x, y) -> (y, 1-x)
            nx = y
            ny = 1 - x
            new_c += [nx, ny]
        new.append((cls, new_c))
    return img_r, new

ROTATE_FUNCS = {
    90: rotate_90,
    180: rotate_180,
    270: rotate_270,
    360: lambda img, lab: (img.copy(), [l for l in lab])
}

def add_noise(img):
    """Thêm nhiễu Gaussian"""
    noise = np.random.randn(*img.shape) * 10
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def blur_image(img, kernel_size=5):
    """Làm mờ ảnh bằng Gaussian Blur"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

print("\nTạo thư mục tạm...")
os.makedirs(tmp_dir / "images", exist_ok=True)
os.makedirs(tmp_dir / "labels", exist_ok=True)

print("\nXử lý ảnh và augmentation...")
tmp_samples = []

img_paths = sorted(list(raw_images_dir.glob("*.jpg")))
print(f"Tìm thấy {len(img_paths)} ảnh trong {raw_images_dir}")

NOISE_LEVELS = [10, 15, 20]

def add_noise_level(img, level):
    noise = np.random.randn(*img.shape) * level
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

for idx, img_path in enumerate(img_paths):
    name = img_path.stem
    lab_path = raw_labels_dir / f"{name}.txt"
    
    if not lab_path.exists():
        print(f"  [BỎ QUA] {name}: Không có file label")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [LỖI] {name}: Không đọc được ảnh")
        continue
    
    labels = load_label(lab_path)
    if len(labels) == 0:
        print(f"  [BỎ QUA] {name}: Không có labels")
        continue

    img_r, lab_r = resize_img_and_label(img, labels, image_size)

    # Tạo thêm n mẫu bằng cách chọn 1 góc xoay random cho mỗi ảnh
    rotation_degrees = [90, 180, 270]
    chosen_rotation = random.choice(rotation_degrees)
    
    batch_1 = []
    # Mẫu gốc (không xoay)
    batch_1.append((f"{name}_orig", img_r.copy(), [l for l in lab_r]))
    # Mẫu xoay random
    img_rot, lab_rot = ROTATE_FUNCS[chosen_rotation](img_r, lab_r)
    batch_1.append((f"{name}_rot{chosen_rotation}", img_rot, lab_rot))
    
    # Nhân đôi để data lên 4n mẫu bằng cách thêm noise với 1 mức random
    batch_2 = []
    chosen_noise_level = random.choice(NOISE_LEVELS)
    
    for sample_name, sample_img, sample_labels in batch_1:
        batch_2.append((sample_name, sample_img, sample_labels))
        img_noise = add_noise_level(sample_img, chosen_noise_level)
        batch_2.append((f"{sample_name}_noise{chosen_noise_level}", img_noise, sample_labels))
    
    # Nhân đôi để data lên 8n mẫu bằng cách chọn 1 augmentation mới
    augmentation_options = ['rot90', 'rot180', 'rot270', 'blur']
    chosen_aug = random.choice(augmentation_options)
    
    final_batch = []
    for sample_name, sample_img, sample_labels in batch_2:
        final_batch.append((sample_name, sample_img, sample_labels))
        
        if chosen_aug == 'rot90':
            img_aug, lab_aug = rotate_90(sample_img, sample_labels)
            final_batch.append((f"{sample_name}_aug90", img_aug, lab_aug))
        elif chosen_aug == 'rot180':
            img_aug, lab_aug = rotate_180(sample_img, sample_labels)
            final_batch.append((f"{sample_name}_aug180", img_aug, lab_aug))
        elif chosen_aug == 'rot270':
            img_aug, lab_aug = rotate_270(sample_img, sample_labels)
            final_batch.append((f"{sample_name}_aug270", img_aug, lab_aug))
        elif chosen_aug == 'blur':
            img_aug = blur_image(sample_img, kernel_size=5)
            final_batch.append((f"{sample_name}_blur", img_aug, sample_labels))
    
    for sample_name, sample_img, sample_labels in final_batch:
        img_path_tmp = tmp_dir / "images" / f"{sample_name}.jpg"
        cv2.imwrite(str(img_path_tmp), sample_img)
        
        lab_path_tmp = tmp_dir / "labels" / f"{sample_name}.txt"
        save_label(lab_path_tmp, sample_labels)
        
        tmp_samples.append(sample_name)
    
    if (idx + 1) % 10 == 0:
        print(f"  Đã xử lý {idx + 1}/{len(img_paths)} ảnh...")

print(f"\nTổng số mẫu sau augmentation: {len(tmp_samples)}")
print("\nChia dataset một cách ngẫu nhiên...")

random.shuffle(tmp_samples)

total = len(tmp_samples)
train_size = int(total * train_ratio)
val_size = int(total * val_ratio)

train_samples = tmp_samples[:train_size]
val_samples = tmp_samples[train_size:train_size + val_size]
test_samples = tmp_samples[train_size + val_size:]

print(f"  Tổng số mẫu: {total}")
print(f"  Train: {len(train_samples)} mẫu ({len(train_samples)/total*100:.1f}%)")
print(f"  Val: {len(val_samples)} mẫu ({len(val_samples)/total*100:.1f}%)")
print(f"  Test: {len(test_samples)} mẫu ({len(test_samples)/total*100:.1f}%)")

print("\nTạo thư mục dataset...")
for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    os.makedirs(dataset_dir / sub, exist_ok=True)

print("\nSao chép file từ tmp sang train/val/test...")

def copy_samples(samples, subset):
    for name in samples:
        src_img = tmp_dir / "images" / f"{name}.jpg"
        dst_img = dataset_dir / f"images/{subset}/{name}.jpg"
        shutil.copy2(src_img, dst_img)
        
        src_lab = tmp_dir / "labels" / f"{name}.txt"
        dst_lab = dataset_dir / f"labels/{subset}/{name}.txt"
        shutil.copy2(src_lab, dst_lab)

copy_samples(train_samples, "train")
copy_samples(val_samples, "val")
copy_samples(test_samples, "test")

print(f"  Đã sao chép {len(train_samples)} mẫu vào train")
print(f"  Đã sao chép {len(val_samples)} mẫu vào val")
print(f"  Đã sao chép {len(test_samples)} mẫu vào test")
print("\nXóa thư mục tạm...")
shutil.rmtree(tmp_dir)
print("  Đã xóa thư mục tmp")
print("\nTạo file data.yaml...")

data_config = {
    "train": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/train"),
    "val": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/val"),
    "test": str("/content/drive/MyDrive/cuoi_ki_ai/dataset/images/test"),
    "nc": len(classes),
    "names": classes
}

with open(data_yaml_path, "w") as f:
    yaml.dump(data_config, f, sort_keys=False)

print(f"\nTạo dataset thành công!")
print(f"  Vị trí: {dataset_dir}")
print(f"  Số lớp: {len(classes)}")
print(f"  Tên các lớp: {classes}")
