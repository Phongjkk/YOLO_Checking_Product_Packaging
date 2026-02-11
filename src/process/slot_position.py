import numpy as np
from utils.caculate import find_collinear_three

def slot_position(slots_boxes):
    """
    NẾU ĐỦ 5 SLOT:
        Vẽ line3 đi qua 3 điểm thẳng hàng, line2 đi qua 2 điểm còn lại.
        Kẻ vector connect(x,y) từ điểm giữa line3 sang line 2, vuông góc với line3.

        Sẽ có 4 trường hợp với line3
        Trường hợp 1: x dương y dương - slot 1 gần góc trên bên phải hơn slot 2 và 3.
        Trường hợp 2: x dương y âm - slot 1 gần gốc toạ độ nhất.
        Trường hợp 3: x âm y âm - slot 1 gần góc dưới bên trái nhất.
        Trường hợp 4: x âm y dương - slot 1 gần góc dưới bên phải nhất.

        Với line2 thì điểm nào gần slot 1 hơn thì sẽ là slot 4, điểm còn lại là slot 5.
    """
    boxes = []
    for slot_pts in slots_boxes:
        center_x = np.mean(slot_pts[:, 0])
        center_y = np.mean(slot_pts[:, 1])
        center_x_ratio = center_x / 640
        center_y_ratio = center_y / 640
        boxes.append([0, center_x_ratio, center_y_ratio])

    centers = np.array([[b[1], b[2]] for b in boxes])

    three_idx = find_collinear_three(centers)
    two_idx = [i for i in range(5) if i not in three_idx]

    line3_pts = centers[three_idx]
    line2_pts = centers[two_idx]

    v_line3 = line3_pts[2] - line3_pts[0]
    perp = np.array([-v_line3[1], v_line3[0]])
    perp = perp / np.linalg.norm(perp)

    p_mid = line3_pts[1]
    vec = line2_pts.mean(axis=0) - p_mid
    vec_connect = perp * np.dot(vec, perp)

    dx, dy = vec_connect

    if dx > 0 and dy > 0:
        corner = np.array([1,0])
    elif dx > 0 and dy < 0:
        corner = np.array([0,0])   
    elif dx < 0 and dy < 0:
        corner = np.array([0,1])
    else:
        corner = np.array([1,1])

    dist3 = np.linalg.norm(line3_pts - corner, axis=1)
    line3_order = [three_idx[i] for i in np.argsort(dist3)]

    slot1_pos = centers[line3_order[0]]
    dist2 = np.linalg.norm(line2_pts - slot1_pos, axis=1)
    line2_order = [two_idx[i] for i in np.argsort(dist2)]

    result = {}
    slot_num = 1

    for idx in line3_order:
        result[slot_num] = [idx, slots_boxes[idx].tolist()]
        slot_num += 1
    for idx in line2_order:
        result[slot_num] = [idx, slots_boxes[idx].tolist()]
        slot_num += 1

    return result
