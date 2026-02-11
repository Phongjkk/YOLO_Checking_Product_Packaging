import json
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = "/content/drive/MyDrive/cuoi_ki_ai"
DATASET_DIR = f"{PROJECT_DIR}/dataset"

try:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
except NameError:
    project_root = Path(PROJECT_DIR)

config_file = project_root / "config" / "config.json"
src_dir = project_root / "src"

with open(config_file, 'r') as f:
    config = json.load(f)
image_size = config["image_size"]
batch_size = config["training_batch_size"]
epochs = config["training_epochs"]

model = YOLO("yolov8n-obb.pt")
model.train(
    data = f"{DATASET_DIR}/data.yaml",
    epochs = epochs,
    imgsz = image_size,
    batch = batch_size,
    device = 0,
    project=f"{PROJECT_DIR}/runs",
    name="model_detect",
)

print("Training xong! Kết quả lưu tại: runs/detect/train")

test_results = model.val(
    data=f"{DATASET_DIR}/data.yaml",
    split="test",
    imgsz=image_size,
    batch=batch_size,
    device=0,
)

result_file = Path(f"{PROJECT_DIR}/runs/model_detect/test_results.json")
with open(result_file, "w") as f:
    json.dump(test_results.results_dict, f, indent=4)
print(f"\nKết quả test đã lưu tại: {result_file}")

predict_dir = Path(f"{PROJECT_DIR}/runs/model_detect/test_images")
predict_dir.mkdir(parents=True, exist_ok=True)

predict_results = model.predict(
    source=f"{DATASET_DIR}/images/test",
    imgsz=image_size,
    device=0,
    save=True,
    project=predict_dir,
    name="",
)
print(f"\nẢnh kết quả test đã lưu tại: {predict_dir}")
