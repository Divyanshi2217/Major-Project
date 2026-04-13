import cv2

prev_x, prev_y = None, None

def draw_line(canvas, x, y, draw=True, erase=False):
    global prev_x, prev_y

    if prev_x is None:
        prev_x, prev_y = x, y
        return

    if draw:
        color = (255, 255, 255) if erase else (0, 0, 0)
        thickness = 20 if erase else 4
        cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)

    prev_x, prev_y = x, y

def reset_line():
    global prev_x, prev_y
    prev_x = prev_y = None
