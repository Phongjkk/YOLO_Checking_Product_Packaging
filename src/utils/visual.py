import cv2
import numpy as np
import time
from collections import deque

class FPSCalculator:
    def __init__(self, buffer_size=30):
        self.timestamps = deque(maxlen=buffer_size)
    
    def update(self):
        self.timestamps.append(time.time())
    
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        return len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0] + 1e-6)

def make_grid(frames):
    target_size = (640, 640)

    black = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    resized_frames = {}
    for name, frame in frames.items():
        resized_frames[name] = cv2.resize(
            frame, target_size, interpolation=cv2.INTER_LINEAR
        )

    f1 = resized_frames.get("cam_1", black)
    f2 = resized_frames.get("cam_2", black)
    f3 = resized_frames.get("cam_3", black)
    f4 = resized_frames.get("cam_4", black)

    top = np.hstack((f1, f2))
    bottom = np.hstack((f3, f4))
    grid = np.vstack((top, bottom))

    return grid

SLOT_COLORS = {
    "empty": (0, 255, 255),      # Vàng
    "oke": (0, 255, 0),          # Xanh lá
    "wrong": (0, 0, 255),        # Đỏ
    "default": (128, 128, 128)   # Xám
}

ITEM_COLOR = (255, 255, 0)    # xanh ngọc

CAM_STATE_COLORS = {
    "done": (0, 255, 0),       # Xanh lá
    "false": (0, 0, 255),      # Đỏ
    "checking": (0, 255, 255), # Vàng
    "waiting": (255, 255, 255)       # Trắng
}

CAM_LABEL_COLOR = (0, 255, 255)
SLOT_LINE_THICKNESS = 3
ITEM_LINE_THICKNESS = 2
TEXT_THICKNESS = 2
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_SCALE_SMALL = 0.5
TEXT_SCALE_LARGE = 0.8

OVERLAY_COLOR = (0, 0, 0)      # Đen
OVERLAY_ALPHA = 0.5            # Độ trong suốt (0-1)
OVERLAY_PADDING = 5            # Khoảng cách padding

def draw_visualization(frame, cam_info, items_boxes):
    _draw_all_slots(frame, cam_info)
    _draw_all_items(frame, items_boxes)
    _draw_camera_info(frame, cam_info)
    _draw_slot_status_table(frame, cam_info)

def _draw_all_slots(frame, cam_info):
    for slot_id, slot in cam_info.slots_list.items():
        points = slot.get_points()
        if points is None:
            continue
        
        pts = np.int32(points)
        color = SLOT_COLORS.get(slot.state, SLOT_COLORS["default"])
        
        # Vẽ box
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=SLOT_LINE_THICKNESS)
        
        # Vẽ label
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        cv2.putText(
            frame,
            f"slot_{slot_id}",
            (center_x - 30, center_y),
            TEXT_FONT,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS
        )

def _draw_all_items(frame, items_boxes):
    """Vẽ tất cả items"""
    for item_name, points in items_boxes:
        pts = np.int32(points)
        
        # Vẽ box
        cv2.polylines(frame, [pts], isClosed=True, color=ITEM_COLOR, thickness=ITEM_LINE_THICKNESS)
        
        # Vẽ label
        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))
        
        cv2.putText(
            frame,
            item_name,
            (center_x - 30, center_y - 10),
            TEXT_FONT,
            TEXT_SCALE,
            ITEM_COLOR,
            TEXT_THICKNESS
        )

def _draw_camera_info(frame, cam_info):
    """Vẽ thông tin camera (label + state) - Màu text theo state"""
    cam_label = _get_camera_label(cam_info.slot_will_be_checked)
    text_color = CAM_STATE_COLORS.get(cam_info.state, CAM_STATE_COLORS["waiting"])
    text = f"{cam_label}: {cam_info.state.upper()}"
    
    # Tính kích thước text
    (text_width, text_height), baseline = cv2.getTextSize(
        text, TEXT_FONT, TEXT_SCALE_LARGE, TEXT_THICKNESS
    )
    
    # Vẽ nền mờ
    x1, y1 = 10 - OVERLAY_PADDING, 30 - text_height - OVERLAY_PADDING
    x2, y2 = 10 + text_width + OVERLAY_PADDING, 30 + baseline + OVERLAY_PADDING
    _draw_overlay(frame, x1, y1, x2, y2)
    
    # Vẽ text
    cv2.putText(
        frame,
        text,
        (10, 30),
        TEXT_FONT,
        TEXT_SCALE_LARGE,
        text_color,
        TEXT_THICKNESS
    )

def _draw_slot_status_table(frame, cam_info):
    """Vẽ bảng trạng thái các slot cần check - Mỗi slot màu riêng theo state"""
    if cam_info.get_state() not in ["checking", "done", "false"]:
        return
    
    # Tính kích thước nền cho toàn bộ bảng
    slot_count = len(cam_info.slot_will_be_checked)
    if slot_count == 0:
        return
    
    # Tính kích thước text mẫu để ước lượng
    sample_text = "S10: wrong"
    (text_width, text_height), baseline = cv2.getTextSize(
        sample_text, TEXT_FONT, TEXT_SCALE_SMALL, TEXT_THICKNESS
    )
    
    # Vẽ nền mờ cho toàn bộ bảng
    x1 = 10 - OVERLAY_PADDING
    y1 = 60 - text_height - OVERLAY_PADDING
    x2 = 10 + text_width + OVERLAY_PADDING * 2
    y2 = 60 + (25 * slot_count) + OVERLAY_PADDING
    _draw_overlay(frame, x1, y1, x2, y2)
    
    # Vẽ từng slot
    y_offset = 60
    for slot_id in cam_info.slot_will_be_checked:
        slot = cam_info.get_slot(slot_id)
        if slot is None:
            continue
        
        # Mỗi slot có màu riêng dựa trên state của nó
        text_color = SLOT_COLORS.get(slot.state, SLOT_COLORS["default"])
        
        cv2.putText(
            frame,
            f"S{slot_id}: {slot.state}",
            (10, y_offset),
            TEXT_FONT,
            TEXT_SCALE_SMALL,
            text_color,
            TEXT_THICKNESS
        )
        y_offset += 25

def _get_camera_label(slot_will_be_checked):
    """Xác định camera label dựa trên slots cần check"""
    slots = set(slot_will_be_checked)
    
    if 1 in slots and 4 not in slots:
        return "CAM_1"
    elif 1 in slots and 4 in slots:
        return "CAM_2"
    elif 6 in slots and 9 not in slots:
        return "CAM_3"
    elif 6 in slots and 9 in slots:
        return "CAM_4"
    else:
        return "CAM_UNKNOWN"

def _draw_overlay(frame, x1, y1, x2, y2):
    """Vẽ nền mờ (overlay) tại vị trí chỉ định"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), OVERLAY_COLOR, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
