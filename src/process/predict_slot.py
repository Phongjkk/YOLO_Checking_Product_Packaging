import numpy as np
from utils.caculate import calculate_deviation, get_transform_params, transform_box, get_box_center

def predict_slot(layout_type, full_box_boxes, slots_boxes):
    """
    Chọn layout
    vòng for chạy 4 lần, mỗi lần khớp 2 góc của full_box_boxes với 2 góc của slots_boxes["full_box"]
        scale cho giá trị tuyệt đối của độ lệch 2 box là thấp nhất
        tính độ lệch của từng slot trong slots_boxes (dùng hàm trên)
        lưu lại lần lặp có tổng độ lệch nhỏ nhất
    lấy lần có tổng độ lệch thấp nhất, thiếu slot mô thì bê từ map sang
    cho lần lượt vào slot_mapping
    """
    layout_map_for_cam_12 = {
        'full_box_boxes': [[[0.5011, 0.0], [0.5011, 0.4536], [0.0, 0.4536], [0.0, 0.0]]],
        'slot_1': [[[0.1757, 0.0473], [0.1778, 0.1457], [0.027, 0.1487], [0.025, 0.0504]]],
        'slot_2': [[[0.1763, 0.1797], [0.1783, 0.278], [0.0275, 0.2811], [0.0255, 0.1827]]],
        'slot_3': [[[0.183, 0.3124], [0.1852, 0.4213], [0.0287, 0.4245], [0.0265, 0.3156]]],
        'slot_4': [[[0.4369, 0.0314], [0.441, 0.2314], [0.243, 0.2354], [0.239, 0.0354]]],
        'slot_5': [[[0.4744, 0.2598], [0.4777, 0.424], [0.2166, 0.4293], [0.2132, 0.2651]]]
    }

    layout_map_for_cam_34 = {
        'full_box_boxes': [[[0.0, 0.0], [0.8917, 0.0], [0.8917, 0.8087], [0.0, 0.8087]]],
        'slot_10': [[[0.4918, 0.0941], [0.7869, 0.0941], [0.7869, 0.2314], [0.4918, 0.2314]]],
        'slot_6': [[[0.118, 0.32], [0.3123, 0.32], [0.3123, 0.7698], [0.118, 0.7698]]],
        'slot_7': [[[0.3917, 0.3261], [0.5474, 0.3261], [0.5474, 0.7738], [0.3917, 0.7738]]],
        'slot_8': [[[0.6593, 0.3149], [0.7764, 0.3149], [0.7764, 0.7474], [0.6593, 0.7474]]],
        'slot_9': [[[0.1122, 0.0564], [0.4103, 0.0564], [0.4103, 0.2742], [0.1122, 0.2742]]]
    }

    if layout_type == 12:
        layout_map = layout_map_for_cam_12
        slot_id_start = 1
    elif layout_type == 34:
        layout_map = layout_map_for_cam_34
        slot_id_start = 6

    layout_map_full_box = np.array(layout_map["full_box_boxes"][0])
    layout_slots = {}
    for i in range(5):
        key = f'slot_{slot_id_start + i}'
        layout_slots[key] = np.array(layout_map[key][0])

    if len(full_box_boxes) == 0:
        return {}, None
    
    detected_full_box = np.array(full_box_boxes[0])
    
    best_match = None
    best_total_deviation = float('inf')
    
    for iteration in range(4):
        layout_corner1 = layout_map_full_box[iteration]
        layout_corner2 = layout_map_full_box[(iteration + 1) % 4]
        
        target_corner1 = detected_full_box[0]
        target_corner2 = detected_full_box[1]
        
        angle, scale, translation = get_transform_params(
            np.array([layout_corner1, layout_corner2]),
            np.array([target_corner1, target_corner2])
        )
        
        transformed_slots = {}
        for key, slot_box in layout_slots.items():
            transformed_slots[key] = transform_box(slot_box, angle, scale, translation)
        
        total_deviation = 0
        for detected_slot in slots_boxes:
            min_deviation = float('inf')
            for key, transformed_slot in transformed_slots.items():
                deviation = calculate_deviation(detected_slot, transformed_slot)
                
                if deviation < min_deviation:
                    min_deviation = deviation
            
            total_deviation += min_deviation
        
        if total_deviation < best_total_deviation:
            best_total_deviation = total_deviation
            best_match = transformed_slots
    
    slot_mapping = {}
    
    if best_match is None:
        return {}, None
    
    detected_centers = []
    for slot_pts in slots_boxes:
        detected_centers.append(get_box_center(slot_pts))
    
    transformed_centers = {}
    for key, slot_box in best_match.items():
        transformed_centers[key] = get_box_center(slot_box)
    
    used_layout_slots = set()
    detected_to_layout = {}
    
    for det_idx, det_center in enumerate(detected_centers):
        min_dist = float('inf')
        closest_key = None
        
        for key, trans_center in transformed_centers.items():
            if key in used_layout_slots:
                continue
            
            dist = np.linalg.norm(det_center - trans_center)
            if dist < min_dist:
                min_dist = dist
                closest_key = key
        
        if closest_key:
            detected_to_layout[det_idx] = closest_key
            used_layout_slots.add(closest_key)
    
    for i in range(5):
        slot_id = slot_id_start + i
        key = f'slot_{slot_id}'
        
        found_detected = False
        for det_idx, layout_key in detected_to_layout.items():
            if layout_key == key:
                slot_mapping[slot_id] = [det_idx, slots_boxes[det_idx].tolist()]
                found_detected = True
                break
        
        if not found_detected:
            slot_mapping[slot_id] = [-1, best_match[key].tolist()]
    
    return slot_mapping, best_match
