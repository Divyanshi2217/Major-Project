import math

def detect_gesture(handLms):
    """Detect gesture based on finger states"""
    fingers = []
    # Thumb
    fingers.append(1 if handLms.landmark[4].x < handLms.landmark[3].x else 0)
    # Other 4 fingers
    for tip in [8, 12, 16, 20]:
        fingers.append(1 if handLms.landmark[tip].y < handLms.landmark[tip - 2].y else 0)

    total = sum(fingers)

    if total == 1:
        return "draw"        # index finger only
    elif total == 2:
        return "line"        # 2 fingers
    elif total == 3:
        return "rectangle"   # 3 fingers
    elif total == 4:
        return "circle"      # 4 fingers
    elif total == 5:
        return "undo"        # all fingers
    elif fingers == [1, 0, 0, 0, 0]:
        return "clear"       # thumb only
    else:
        return None
