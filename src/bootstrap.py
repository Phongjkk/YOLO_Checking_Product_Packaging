import json
import torch # Để kiểm tra GPU trên Windows

from camera.CamThread import CamThread
from camera.CamInfo import SlotInfo, CamInfo

def bootstrap(config_file):
    # 1. Đọc cấu hình từ file JSON
    with open(config_file, "r") as f:
        config = json.load(f)

    image_size = config["image_size"]
    classes = config["classes"]
    url_1 = config["url_1"]
    url_2 = config["url_2"]
    url_3 = config["url_3"]
    url_4 = config["url_4"]

    # 2. Khởi tạo danh sách các Slot (Vị trí linh kiện)
    slot_expected_items = {
        i: config[f"slot_{i}"] for i in range(1, 11)
    }

    slots = {
        i: SlotInfo(expected_item=slot_expected_items[i])
        for i in range(1, 11)
    }

    # Chia slot cho các cụm Camera (Cam 1-2 soi nửa đầu, Cam 3-4 soi nửa sau)
    slots_list_for_cam_12 = {i: slots[i] for i in range(1, 6)}
    slots_list_for_cam_34 = {i: slots[i] for i in range(6, 11)}

    # 3. Định nghĩa logic kiểm tra cho từng Camera
    cameras = {
        "cam_1": CamInfo(slot_will_be_checked=[1, 2, 3], slots_list=slots_list_for_cam_12),
        "cam_2": CamInfo(slot_will_be_checked=[1, 2, 3, 4, 5], slots_list=slots_list_for_cam_12),
        "cam_3": CamInfo(slot_will_be_checked=[6, 7, 8], slots_list=slots_list_for_cam_34),
        "cam_4": CamInfo(slot_will_be_checked=[6, 7, 8, 9, 10], slots_list=slots_list_for_cam_34)
    }

    cam_configs = [
        ("cam_1", url_1),
        ("cam_2", url_2),
        ("cam_3", url_3),
        ("cam_4", url_4),
    ]

    # 4. Khởi tạo các luồng Camera với GPU Acceleration cho Windows
    # Lưu ý: gpu_type="nvidia" sẽ kích hoạt CUDA trong FFmpeg
    cam_threads = {
        name: CamThread(name, source, mode="rtsp", gpu_type="nvidia")
        for name, source in cam_configs
    }

    # 5. Thiết lập thiết bị xử lý AI (Inference Device)
    # Trên Windows, dùng CUDA nếu có card NVIDIA, ngược lại dùng CPU
    if torch.cuda.is_available():
        device = '0' # Tương đương cuda:0
        print(f"[System] GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("[System] No GPU found, falling back to CPU.")

    return image_size, classes, cameras, cam_threads, device

# --- Giải thích các thay đổi cho Windows ---
# - Đổi 'mps:0' thành '0' (CUDA mặc định).
# - Thêm tham số 'gpu_type="nvidia"' cho CamThread để dùng hardware decoding.
# - Thêm logic tự động kiểm tra CUDA khả dụng.