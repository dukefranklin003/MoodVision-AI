import os
import joblib
import numpy as np
import cv2

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_emotion_model.pkl")

SMOOTHING_WINDOW = 7
recent_preds = []

def smooth_prediction(pred_idx):
    recent_preds.append(pred_idx)
    if len(recent_preds) > SMOOTHING_WINDOW:
        recent_preds.pop(0)
    counts = np.bincount(recent_preds)
    return int(np.argmax(counts))

def main():
    # Load best model
    if not os.path.exists(BEST_MODEL_PATH):
        print("Best model not found. Train the model first.")
        return

    state = joblib.load(BEST_MODEL_PATH)
    model = state["model"]
    scaler = state["scaler"]
    best_val_acc = state.get("best_val_acc", 0.0)
    classes = state.get("classes", model.classes_)
    img_size = state.get("img_size", 64)

    print(f"Loaded best model from {BEST_MODEL_PATH} (val_acc={best_val_acc:.4f})")
    print("Classes:", classes)

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Failed to load Haar cascade.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Mirror effect so you feel like a mirror
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]

                # Preprocess like training
                face_resized = cv2.resize(face_roi, (img_size, img_size))
                vec = face_resized.astype("float32").flatten() / 255.0
                vec = vec.reshape(1, -1)

                # Scale with the same scaler used in training
                vec_scaled = scaler.transform(vec)

                # Predict emotion
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec_scaled)[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                else:
                    pred_idx = int(model.predict(vec_scaled)[0])
                    confidence = 0.0

                smooth_idx = smooth_prediction(pred_idx)

                emotion_raw = classes[pred_idx]
                emotion_smooth = classes[smooth_idx]

                # Draw bounding box & label
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)

                label = f"{emotion_smooth}"
                if confidence > 0:
                    label += f" ({confidence*100:.1f}%)"

                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)

        cv2.imshow("Real-time Emotion Detection (Classic ML)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
