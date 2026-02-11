"""
QUY TRÌNH ĐÓNG GÓI:
    - Cam 1: Cho 3 "mach_nho" vào slot_1, slot_2 và slot_3.                             
    - Cam 2: Cho "mach_lon" vào slot_4, cho usb_to_jtag vào slot_5                      
    - Cam 3: Cho "day_black" vào slot_6, "day_lgbt" vào slot_7, "day_white" vào slot_8  
    - Cam 4: Cho "pack_circut" vào slot_9, "day_gray" vào slot_10                       

QUY TRÌNH XỬ LÝ:
    - Ban đầu, trạng thái của cam là waiting, chạy mô hình yolo.
    - Khi 5 slot xuất hiện, chuyển trạng thái sang checking và 
      chạy hàm slot_position để xác định slot nào là slot 1,2,3,4,5.
    - Khi 1 item xuất hiện và va chạm với 1 slot, chạy hàm check_item_and_slot.
      (hàm này sẽ nhận vào items, số thứ tự slot, toạ độ item, toạ độ slot, config. Nếu item 80% item nằm trong slot thì xác định item đã ở trong slot, 
      check config xem chúng có thuộc về nhau không, và trả về "true" hoặc "false").
    - Nếu "true", chuyển slot.state sang "oke", nếu "false" thì "wrong".
    - Khi tất cả các slot_will_be_checked trong cam có state "oke" thì cam.state được chuyển sang "done", 
      ngược lại nếu có ít nhất một slot.state là "wrong" thì chuyển cam.state sang "false".

CLASS SlotInfo:
    - Gồm id, expected(string), points (mảng np có 4 điểm xác định phạm vi slot), state(string)
    - Hàm set_state(), set_points()
    - Hàm get_state(), get_points()
    - Hàm update()
    
CLASS CamInfo:
    - Gồm cam_id, slot_will_be_checked, state, slots
    - Hàm init(id) dùng để khởi tạo thông tin cam (id=id, slot_will_be_checked lấy trong config, state="waiting", slots rỗng) trong file detect.py
    - Hàm set_state(new_state) dùng để thay đổi state của cam
    - Hàm add_slot()
    - Hàm get_slot()

WORK FLOW FILE detect.py:
    - 1. Mở 4 camera bằng Camthread
    - 2. Khởi tạo riêng biệt thông tin của mỗi camera bằng CamInfo.init()
    - 3. Khởi tạo YOLOProcesser
    - 4. Khởi tạo detection_results để lưu kết quả cuối cùng nhận được từ Yolo
    - 5. Chạy vòng lặp chính

MAIN LOOP TRONG FILE detect.py:
    - Đọc frames từ 4 camera đã mở từ bước 1
    - Gom 4 frames thành 1 batch, gửi cho YOLOProcesser xử lý
    - Lấy kết quả từ YOLOProcesser (object slot là 0, 1->8 là 8 items)
    - Nếu detection_results không trống, gọi hàm process_yolo_results để xử lý frame và
    - Tạo grid 2x2 và hiển thị frames

HÀM process_results_from_yolo:
    - Nhận đầu vào là frames (4 frame), detection_results và thông tin của 4 camera ở bước 2 bên trên
    - Tạo thread xử lý song song 4 camera cùng lúc (hàm process_single_camera cho mỗi cam)
    - Hàm process_single_camera:
        - Đầu vào là cam_id, frame và detection_results của cam đó, caminfo
        - Đầu tiên, kiểm tra detection_results:
            - Nếu không có đúng 5 slot xuất hiện, bỏ qua không xử lý gì cả
            - Nếu có 5 đúng slot, chạy hàm slot_position(), cập nhật info cho cả 5 slot
            - Dùng hàm set_state("checking")
        - Nếu trong detection_results có một item nào đó (1->8) thì chạy hàm is_item_in_slot_and_valid(), hàm này sẽ trả về true/false
        - Nếu "true" thì dùng set_state("oke"), "false" thì set_state("wrong")
        - If tất cả slot trong slot_will_be_checked có state "oke" (dùng get_state()) thì chuyển state của cam sang "done" (get_state("done")). Else get_state("false)
    - Vẽ lên màn hình các box mà detection_results có, hiển thị cam.state và các cam.slots.state (chỉ các slot có trong danh sách slot_will_be_checked)

NÂNG CẤP cải thiện FPS:
    - Ban đầu cả 4 cam đều đang "waiting" gửi cả 4 frame cho yolo detect.
    - Khi một camera chuyển sang state != "waiting" thì chỉ gửi frame đó cho yolo thôi, nó là active_cam.
    - Khi active_cam chuyển state về "waiting" (trong 5 frames liên tiếp) thì quay lại gửi batch 4 frame để chờ active_cam mới xuất hiện.
    - Đề phòng trường hợp có 2 cam cùng chuyển state thì lấy cam đứng trước
"""
import cv2
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

from bootstrap import bootstrap
from utils.visual import make_grid
from process.process_results_from_yolo import process_results_from_yolo, center_crop_and_resize

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
config_file = project_root / "config" / "config.json"
model_file = project_root / "model" / "best.pt"

image_size, classes, cameras, cam_threads, device = bootstrap(config_file)

model = YOLO(model_file).to('cuda')
model.fuse()

executor = ThreadPoolExecutor(max_workers=4)

active_cam = None
waiting_counter = {}

while True:
    frames = {}
    resized_frames = {}

    for name, cam in cam_threads.items():
        frame, fps = cam.read()
        if frame is not None:
            frames[name] = frame
            resized_frames[name] = center_crop_and_resize(frame, image_size)

    if len(frames) == 0:
        continue

    cam_names = list(frames.keys())
    
    if active_cam is None:
        frame_list = [resized_frames[n] for n in cam_names]
        process_cam_names = cam_names
    else:
        frame_list = [resized_frames[active_cam]]
        process_cam_names = [active_cam]

    batch_results = model(
        frame_list,
        imgsz=image_size,
        conf=0.4,
        device=device,
        verbose=False
    )

    frames = process_results_from_yolo(
        frames=resized_frames,              
        batch_results=batch_results,
        cameras=cameras,
        cam_names=process_cam_names,
        classes=classes,
        executor=executor
    )

    if active_cam is not None: # đang ở mode 1 cam
        if cameras[active_cam].state == "waiting":
            waiting_counter[active_cam] = waiting_counter.get(active_cam, 0) + 1
            if waiting_counter[active_cam] >= 5:
                active_cam = None
                waiting_counter = {}
        else:
            waiting_counter[active_cam] = 0

    else: # đang ở mode batch
        for name in cam_names:
            if name in cameras and cameras[name].state != "waiting":
                active_cam = name
                waiting_counter = {n: 0 for n in cam_names}
                break

    grid = make_grid(frames)
    cv2.imshow("Checking product packaging", grid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n[INFO] Shutting down...")
for cam in cam_threads.values():
    cam.release()

executor.shutdown(wait=True)
cv2.destroyAllWindows()