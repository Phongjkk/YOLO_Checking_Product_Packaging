import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class SlotInfo:
    expected_item: str
    points: Optional[np.ndarray] = None
    state: str = "empty" # empty / oke / wrong

    def __init__(self, expected_item, points=None, state="empty"):
        self.expected_item = expected_item
        self.points = points
        self.state = state

    def set_state(self, new_state: str):
        self.state = new_state

    def set_points(self, new_points: Optional[np.ndarray]):
        self.points = new_points

    def get_state(self) -> str:
        return self.state

    def get_points(self) -> Optional[np.ndarray]:
        return self.points
    
    def get_expected_item(self):
        return self.expected_item

    def update(self, points: Optional[np.ndarray]):
        self.points = points

@dataclass
class CamInfo:
    slot_will_be_checked: List[int]
    slots_list: Dict[int, SlotInfo]
    state: str = "waiting" # waiting / checking / done / false
    
    def __init__(self, slot_will_be_checked, slots_list, state="waiting"):
        self.state = state
        self.slot_will_be_checked = slot_will_be_checked
        self.slots_list = slots_list

    def set_state(self, new_state: str):
        self.state = new_state

    def get_state(self) -> str:
        return self.state

    def get_slot(self, slot_id: int) -> Optional[SlotInfo]:
        return self.slots_list.get(slot_id)

    def update_slot(self, slot_mapping):
        # predict_slot trả về (slot_mapping, best_match)
        if isinstance(slot_mapping, tuple): 
            slot_mapping = slot_mapping[0]

        offset = 0
        if self.slot_will_be_checked and self.slot_will_be_checked[0] >= 6:
            offset = 5

        for slot_num, value in slot_mapping.items():
            adjusted_slot_id = slot_num + offset
            slot = self.get_slot(adjusted_slot_id)
            if slot is None:
                continue

            # value = [det_idx, points]
            if (
                isinstance(value, (list, tuple)) and
                len(value) == 2
            ):
                points = np.array(value[1]) if value[1] is not None else None
            else:
                points = None

            slot.update(points)
