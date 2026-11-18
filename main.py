import cv2
import numpy as np

# ---------- SETUP ----------
NAMES = ["person", "chair", "book"]

MIN_SCORE = 0.5

# Load AI object detection model
model = ("yolov8n.pt")

# Start camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Press Q to quit.")

# ---------- DETECTION DATA ----------
OUTPUT = np.array([
    [0.90, 0.90, 0.05, 0.05],
    [0.80, 0.10, 0.80, 0.10],
    [0.60, 0.20, 0.10, 0.70],
    [0.50, 0.60, 0.30, 0.10],
    [0.10, 0.90, 0.05, 0.05],
    [0.70, 0.30, 0.20, 0.90],
])

scores = OUTPUT[:, 0:1] * OUTPUT[:, 1:]

print("\nPART 1: Scores")
print("Shape:", scores.shape)
print("Guess 4, person:", round(float(scores[4, 0]), 2))

# ---------- BEST NAME ----------
class_ids = scores.argmax(axis=1)
confidences = scores.max(axis=1)

print("\nPART 2: Best name")

for i in range(6):
    print("Guess", i, "->", NAMES[class_ids[i]],
          f"{confidences[i]:.2f}")

# ---------- FILTER GUESSES ----------
kept = [i for i in range(6) if confidences[i] >= MIN_SCORE]

print("\nPART 3: Kept")
print("Kept guesses:", [int(i) for i in kept])
print("Kept names:", [NAMES[class_ids[i]] for i in kept])

# ---------- THRESHOLDS ----------
print("\nPART 4: Three thresholds")

for threshold in (0.2, 0.5, 0.8):
    count = int((confidences >= threshold).sum())
    print("Threshold", threshold, "->", count, "kept")

# ---------- SHOW CAMERA ----------
while True:

    # ---------- CAMERA ----------
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read camera.")
        break

    results = model(frame, verbose=False)

    all_scores = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = model.names[class_id]

            if name not in NAMES:
                continue

            # Store prediction
            all_scores.append((name, confidence))

            # Draw detection if above threshold
            if confidence >= MIN_SCORE:

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0]
                )
                

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{name}: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Convert predictions into NumPy scores
    scores = np.array([
        confidence for name, confidence in all_scores
    ])

    # ---------- PART 1: SCORES ----------
    print("\nPART 1: Scores")

    if len(scores) > 0:
        print("Shape:", scores.shape)
        print("Scores:", np.round(scores, 2))

    # ---------- PART 2: BEST NAME ----------
    print("\nPART 2: Best name")

    if len(scores) > 0:
        class_ids = np.array([
            NAMES.index(name) for name, confidence in all_scores
        ])

        for i, (name, confidence) in enumerate(all_scores):
            print(
                "Guess", i,
                "->", name,
                f"{confidence:.2f}"
            )

    # ---------- PART 3: KEPT ----------
    print("\nPART 3: Kept")

    kept = [
        i for i in range(len(all_scores))
        if all_scores[i][1] >= MIN_SCORE
    ]

    print("Kept guesses:", kept)
    print("Kept names:", [
        all_scores[i][0] for i in kept
    ])

    # ---------- PART 4: THREE THRESHOLDS ----------
    print("\nPART 4: Three thresholds")

    for threshold in (0.2, 0.5, 0.8):
        count = sum(
            confidence >= threshold
            for name, confidence in all_scores
        )

        print(
            "Threshold", threshold,
            "->", count, "kept"
        )

    # Display camera
    cv2.imshow("AI Object Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close camera
cap.release()
cv2.destroyAllWindows()