import os
import glob
import re
import joblib
import numpy as np
import cv2

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# ================= CONFIG =================
DATA_DIR = "data"
IMG_SIZE = 64              # final face size: 64x64
TOTAL_EPOCHS = 30          # you can increase later
BATCH_SIZE = 64            # mini-batch size for partial_fit

MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_emotion_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ==========================================


def load_dataset(data_dir, img_size=64):
    """Load all images, detect face, crop, resize, flatten."""
    X = []
    y = []

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Warning: could not load Haar cascade. Will resize full images.")

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Loading class: {class_name}")
        for fname in os.listdir(class_path):
            fpath = os.path.join(class_path, fname)
            if not os.path.isfile(fpath):
                continue

            img = cv2.imread(fpath)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Face detection
            face_roi = None
            if not face_cascade.empty():
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    # pick the largest face
                    x, y0, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face_roi = gray[y0:y0 + h, x:x + w]

            if face_roi is None:
                # fallback: use full image
                face_roi = gray

            # Resize to fixed size
            face_resized = cv2.resize(face_roi, (img_size, img_size))

            # Normalize to [0,1] and flatten
            vec = face_resized.astype("float32").flatten() / 255.0
            X.append(vec)
            y.append(class_name)

            # Simple augmentation: horizontally flipped face
            flipped = cv2.flip(face_resized, 1)
            vec_flip = flipped.astype("float32").flatten() / 255.0
            X.append(vec_flip)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)
    return X, y


def get_latest_checkpoint(checkpoint_dir):
    """
    Return (model, scaler, last_epoch, best_val_acc) if a valid checkpoint exists.
    Ignore older/incompatible checkpoints that don't contain 'scaler'.
    """
    pattern = os.path.join(checkpoint_dir, "epoch_*.pkl")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None, None, 0, 0.0

    epoch_ckpts = []
    for path in ckpts:
        m = re.search(r"epoch_(\d+)\.pkl", os.path.basename(path))
        if m:
            epoch_num = int(m.group(1))
            epoch_ckpts.append((epoch_num, path))

    if not epoch_ckpts:
        return None, None, 0, 0.0

    # Sort newest last
    epoch_ckpts.sort(key=lambda x: x[0])

    # Iterate from latest backwards, pick the first that has both model & scaler
    for last_epoch, last_path in reversed(epoch_ckpts):
        try:
            state = joblib.load(last_path)
            model = state.get("model", None)
            scaler = state.get("scaler", None)
            best_val_acc = state.get("best_val_acc", 0.0)
            if model is not None and scaler is not None:
                print(f"Using checkpoint: {last_path}")
                return model, scaler, last_epoch, best_val_acc
            else:
                print(f"Ignoring old/incomplete checkpoint: {last_path}")
        except Exception as e:
            print(f"Failed to load checkpoint {last_path}: {e}")

    # If none usable, start fresh
    return None, None, 0, 0.0


def main():
    print(">>> Loading dataset...")
    X, y = load_dataset(DATA_DIR, IMG_SIZE)

    if len(X) == 0:
        print("No images found in data folder. Please check DATA_DIR.")
        return

    print("Total samples (including augmentation):", len(X))
    print("Emotion classes:", sorted(set(y)))

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    classes = np.unique(y_train)

    # Compute class weights for 'balanced' training
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = {cls: w for cls, w in zip(classes, class_weights_array)}
    print("Class weights:", class_weight_dict)

    # Try to resume from checkpoint
    model, scaler, last_epoch, best_val_acc = get_latest_checkpoint(CHECKPOINT_DIR)
    first_epoch_for_model = False

    if model is None or scaler is None:
        print("No valid checkpoint found. Initializing new scaler and model.")

        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = SGDClassifier(
            loss="log_loss",          # logistic regression with probabilities
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            max_iter=1,               # we'll loop epochs manually
            warm_start=True,
            random_state=42
        )

        first_epoch_for_model = True
        start_epoch = 1
        best_val_acc = 0.0
    else:
        print(f"Resuming from epoch {last_epoch + 1}")
        print(f"Best validation accuracy so far: {best_val_acc:.4f}")
        # Recompute scaled features using loaded scaler
        X_train_scaled = scaler.transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        start_epoch = last_epoch + 1

    # ========== Training loop ==========
    n_train = len(X_train_scaled)

    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{TOTAL_EPOCHS} ===")

        # Shuffle training indices
        indices = np.random.permutation(n_train)
        X_epoch = X_train_scaled[indices]
        y_epoch = y_train[indices]

        # Mini-batch training
        offset = 0
        while offset < n_train:
            end = min(offset + BATCH_SIZE, n_train)
            X_batch = X_epoch[offset:end]
            y_batch = y_epoch[offset:end]

            # Build sample_weight using class weights
            sample_weight = np.array([class_weight_dict[label] for label in y_batch])

            if first_epoch_for_model:
                model.partial_fit(X_batch, y_batch, classes=classes, sample_weight=sample_weight)
                first_epoch_for_model = False
            else:
                model.partial_fit(X_batch, y_batch, sample_weight=sample_weight)

            offset = end

        # Evaluate on training and validation
        y_train_pred = model.predict(X_train_scaled)
        y_val_pred = model.predict(X_val_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Val accuracy:   {val_acc:.4f}")

        # Save checkpoint for this epoch
        state = {
            "epoch": epoch,
            "model": model,
            "scaler": scaler,
            "best_val_acc": best_val_acc
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pkl")
        joblib.dump(state, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": model,
                "scaler": scaler,
                "best_val_acc": best_val_acc,
                "classes": classes,
                "img_size": IMG_SIZE
            }
            joblib.dump(best_state, BEST_MODEL_PATH)
            print(f"New BEST model saved (val_acc={best_val_acc:.4f}) at {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
