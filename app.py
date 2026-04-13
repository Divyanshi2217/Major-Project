import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.pagesizes import letter

# ===================== MEDIAPIPE INIT =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ===================== SCREEN =====================
screen_w, screen_h = 1280, 720
canvas = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255

# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# ===================== DRAW SETTINGS =====================
prev_x, prev_y = None, None
smooth_x, smooth_y = 0, 0
alpha = 0.6
brush_thickness = 7
erase_thickness = 40

# ===================== BUTTONS =====================
clear_btn = (50, screen_h - 100, 250, screen_h - 40)
pdf_btn = (300, screen_h - 100, 650, screen_h - 40)

# ===================== STATUS =====================
status_text = ""
status_timer = 0


# ===================== DRAW BUTTON =====================
def draw_button(img, x1, y1, x2, y2, text):
    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 200), 2)

    cv2.putText(img, text, (x1 + 20, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)


# ===================== SAVE TO PDF =====================
def save_to_pdf(canvas_img):
    global status_text, status_timer

    image_path = "temp_canvas.png"
    cv2.imwrite(image_path, canvas_img)

    filename = f"Handwriting_{datetime.now().strftime('%H%M%S')}.pdf"

    pdf = pdf_canvas.Canvas(filename, pagesize=letter)
    pdf.drawString(200, 750, "Handwritten Notes")

    pdf.drawImage(image_path, 50, 200, width=500, height=400)

    pdf.save()

    status_text = f"PDF Saved Successfully: {filename}"
    status_timer = time.time()


# ===================== MOUSE CLICK =====================
def mouse_click(event, x, y, flags, param):
    global canvas

    if event == cv2.EVENT_LBUTTONDOWN:

        # Clear canvas
        if clear_btn[0] <= x <= clear_btn[2] and clear_btn[1] <= y <= clear_btn[3]:
            canvas[:] = 255

        # Save PDF
        elif pdf_btn[0] <= x <= pdf_btn[2] and pdf_btn[1] <= y <= pdf_btn[3]:
            save_to_pdf(canvas)


# ===================== WINDOW =====================
cv2.namedWindow("AI Virtual Writing", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Virtual Writing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("AI Virtual Writing", mouse_click)

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * screen_w)
            y = int(hand_landmarks.landmark[8].y * screen_h)

            index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

            palm_open = index_up and middle_up and pinky_up

            smooth_x = int(alpha * smooth_x + (1 - alpha) * x)
            smooth_y = int(alpha * smooth_y + (1 - alpha) * y)

            # Draw
            if index_up and not middle_up:
                if prev_x is None:
                    prev_x, prev_y = smooth_x, smooth_y

                cv2.line(canvas, (prev_x, prev_y),
                         (smooth_x, smooth_y),
                         (0, 0, 0),
                         brush_thickness)

                prev_x, prev_y = smooth_x, smooth_y

            # Erase
            elif palm_open:
                cv2.circle(canvas, (smooth_x, smooth_y),
                           erase_thickness, (255, 255, 255), -1)
                prev_x, prev_y = None, None

            else:
                prev_x, prev_y = None, None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        prev_x, prev_y = None, None

    # Header UI
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (screen_w, 60), (30, 30, 30), -1)
    canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

    cv2.putText(canvas, "AI Virtual Writing System",
                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 2)

    # Webcam preview
    cam_small = cv2.resize(frame, (240, 180))
    canvas[20:200, screen_w - 260:screen_w - 20] = cam_small
    cv2.rectangle(canvas, (screen_w - 260, 20),
                  (screen_w - 20, 200), (0, 255, 200), 2)

    # Buttons
    draw_button(canvas, *clear_btn, "Clear Canvas")
    draw_button(canvas, *pdf_btn, "Convert to PDF")

    # Status message
    if time.time() - status_timer < 3:
        cv2.putText(canvas, status_text,
                    (650, screen_h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 200, 0),
                    2)

    cv2.imshow("AI Virtual Writing", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()