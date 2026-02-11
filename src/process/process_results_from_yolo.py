import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from utils.visual import draw_visualization
from process.slot_position import slot_position
from process.predict_slot import predict_slot
from camera.CamInfo import CamInfo
from utils.caculate import is_item_in_slot, is_valid_item

# ==========================================================
# HÀM BỔ SUNG ĐỂ SỬA LỖI IMPORT
# ==========================================================
def center_crop_and_resize(frame, target_size=640):
    """
    Cắt khung hình thành hình vuông từ tâm và resize về target_size.
    Giúp hình ảnh không bị méo khi đưa vào YOLO.
    """
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    
    # Tính toán tọa độ cắt từ tâm
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Cắt thành hình vuông
    crop_img = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    
    # Resize về kích thước chuẩn (ví dụ 640x640)
    resized_img = cv2.resize(crop_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    return resized_img

# ==========================================================
# LOGIC XỬ LÝ CHO TỪNG CAMERA
# ==========================================================
def process_single_camera(yolo_results: dict, frame: np.ndarray, cam_info: CamInfo):
    full_box_boxes = yolo_results['full_box_boxes']
    slots_boxes = yolo_results['slots_boxes']
    items_boxes = yolo_results['items_boxes']

    # Nếu không thấy slot nào -> Reset trạng thái chờ
    if len(slots_boxes) == 0:
        cam_info.set_state("waiting")
        for slot_id, slot in cam_info.slots_list.items():
            slot.set_state("empty")
            slot.set_points(None)
    else:
        if cam_info.get_state() == "waiting":
            cam_info.set_state("checking")

        # Logic xác định vị trí Slot
        if len(slots_boxes) < 5:
            # Nếu thiếu slot, dùng predict_slot để đoán vị trí dựa trên layout mẫu
            layout_type = 12 if min(cam_info.slot_will_be_checked) == 1 else 34
            cam_info.update_slot(predict_slot(layout_type, full_box_boxes, slots_boxes))
        elif len(slots_boxes) == 5:
            # Nếu đủ 5, dùng slot_position để gán ID dựa trên hình học
            cam_info.update_slot(slot_position(slots_boxes))

        # Kiểm tra vật phẩm trong từng slot
        for slot_id in cam_info.slot_will_be_checked:
            slot = cam_info.get_slot(slot_id)
            if slot is None: continue
            
            slot_points = slot.get_points()
            found_item_in_slot = False

            for item_name, item_points in items_boxes:
                if slot_points is not None and is_item_in_slot(item_points, slot_points):
                    found_item_in_slot = True
                    # Check xem item có đúng loại quy định không
                    state = "oke" if is_valid_item(item_name, slot.expected_item) else "wrong"
                    slot.set_state(state)
                    break # Tìm thấy item cho slot này rồi thì sang slot tiếp theo

            if not found_item_in_slot:
                slot.set_state("empty")

        # Cập nhật trạng thái tổng quát của Camera
        all_slots = [cam_info.get_slot(sid) for sid in cam_info.slot_will_be_checked]
        states = [s.get_state() for s in all_slots if s is not None]

        if all(st == "oke" for st in states):
            cam_info.set_state("done")
        elif any(st == "wrong" for st in states):
            cam_info.set_state("false")
        else:
            cam_info.set_state("checking")

    # Vẽ kết quả lên frame
    draw_visualization(frame, cam_info, items_boxes)

# ==========================================================
# ĐIỀU PHỐI KẾT QUẢ TỪ BATCH YOLO
# ==========================================================
def process_results_from_yolo(frames: dict, batch_results: list, cameras: dict, cam_names: list, classes: list, executor: ThreadPoolExecutor) -> dict:
    yolo_results = {}

    for idx, cam_id in enumerate(cam_names):
        res = batch_results[idx]
        full_box_boxes, slots_boxes, items_boxes = [], [], []

        if res.obb is not None:
            # Chuyển dữ liệu từ GPU sang CPU để xử lý logic numpy
            xyxyxyxy = res.obb.xyxyxyxy.cpu().numpy()
            cls_ids = res.obb.cls.cpu().numpy().astype(int)
            confs = res.obb.conf.cpu().numpy()

            for pts, cls_id, score in zip(xyxyxyxy, cls_ids, confs):
                pts = pts.astype(np.float32)

                # Ngưỡng tin cậy (Confidence threshold)
                if cls_id == 0 and score < 0.4: continue # Slot
                if cls_id >= 1 and cls_id != 9 and score < 0.5: continue # Items

                if cls_id == 0: # Slot
                    if len(slots_boxes) < 5:
                        slots_boxes.append((score, pts))
                    else:
                        # Chỉ giữ lại 5 slot có điểm cao nhất
                        min_idx = min(range(5), key=lambda i: slots_boxes[i][0])
                        if score > slots_boxes[min_idx][0]:
                            slots_boxes[min_idx] = (score, pts)
                            
                elif cls_id == 9: # Full Box (Vỏ hộp)
                    if len(full_box_boxes) == 0 or score > full_box_boxes[0][0]:
                        full_box_boxes = [(score, pts)]
                else: # Các loại Item khác
                    item_name = classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}"
                    items_boxes.append((item_name, pts))

        yolo_results[cam_id] = {
            "full_box_boxes": [pts for _, pts in full_box_boxes],
            "slots_boxes": [pts for _, pts in slots_boxes],
            "items_boxes": items_boxes
        }

    # Chạy đa luồng xử lý 4 camera song song
    futures = []
    for cam_name in cam_names:
        frame = frames.get(cam_name)
        cam_info = cameras.get(cam_name)
        if frame is not None and cam_info is not None:
            futures.append(
                executor.submit(process_single_camera, yolo_results[cam_name], frame, cam_info)
            )

    for future in futures:
        future.result() # Đợi các luồng xử lý xong

    return frames