from ultralytics import YOLO
import cv2
import winsound


# 1. LOAD YOLO MODEL

model = YOLO("yolov8s.pt")


# 2. OPEN VIDEO

cap = cv2.VideoCapture("crowd_vedio.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# 3. SETTINGS

CROWD_LIMIT = 5
CONFIDENCE = 0.6

count_history = []
alarm_on = False


# 4. PROCESS VIDEO

while True:

    ret, frame = cap.read()

    if not ret:
        break


    # YOLO DETECTION + TRACKING

    results = model.track(
        frame,
        conf=CONFIDENCE,
        iou=0.45,
        imgsz=960,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    person_count = 0


    # PROCESS DETECTIONS

    for result in results:

        if result.boxes is None:
            continue

        # Only PERSON class
        # COCO class 0 = person
        person_boxes = result.boxes[
            result.boxes.cls == 0
        ]

        # Additional confidence filtering
        person_boxes = person_boxes[
            person_boxes.conf > CONFIDENCE
        ]

        person_count = len(person_boxes)


        # DRAW EACH PERSON

        for box in person_boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Slightly tighten bounding box
            pad = 5

            x1 += pad
            y1 += pad
            x2 -= pad
            y2 -= pad


            # GET TRACKING ID

            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1


            # DRAW BOUNDING BOX

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # DISPLAY TRACKING ID

            label = f"Person ID: {track_id}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # SMOOTH CROWD COUNT

    count_history.append(person_count)

    if len(count_history) > 10:
        count_history.pop(0)

    stable_count = int(
        sum(count_history) / len(count_history)
    )


    # CROWD ALERT

    if stable_count > CROWD_LIMIT:

        if not alarm_on:
            winsound.Beep(1200, 700)
            alarm_on = True

        cv2.putText(
            frame,
            "CROWD ALERT!",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    else:
        alarm_on = False


    # DISPLAY CROWD COUNT

    cv2.putText(
        frame,
        f"Crowd Count: {stable_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )


    # SHOW VIDEO

    cv2.imshow(
        "YOLO Crowd Detection + Tracking",
        frame
    )

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# 5. CLEANUP

cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")
