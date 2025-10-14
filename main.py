from collections import deque
import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from constants import HISTORY_LENGTH
from utils.classifier import compute_hog_features, preprocess_hand
from utils.gui import create_trackbars, get_values_from_trackbars
from utils.video import (
    display_combined,
    draw_prediction,
    extract_best_contour,
    find_valid_contours,
    get_cropped_bin,
    load_pipeline,
    preprocess_frame,
)


def video():
    scaler, pca, model = load_pipeline()
    history = deque(maxlen=HISTORY_LENGTH)
    cap = cv2.VideoCapture(0)
    create_trackbars()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        lower, upper = get_values_from_trackbars()
        mask = preprocess_frame(frame, lower, upper)
        contours = find_valid_contours(mask)

        if not contours:
            # Nothing detected — show default empty hand frame
            display_combined(frame, mask, np.zeros_like(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        best = extract_best_contour(contours, mask, frame, model, scaler, pca)

        if best is None:
            # No sufficiently large contour — skip classification
            display_combined(frame, mask, np.zeros_like(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        contour, confidence, pred_class = best
        cropped_bin_filled = get_cropped_bin(frame, mask, contour)

        history.append(pred_class)
        draw_prediction(frame, pred_class, history, HISTORY_LENGTH)

        cropped = cv2.cvtColor(cropped_bin_filled, cv2.COLOR_GRAY2BGR)
        scale = frame.shape[0] / cropped.shape[0]
        new_w = int(cropped.shape[1] * scale)
        resized = cv2.resize(cropped, (new_w, frame.shape[0]))
        hand = np.zeros_like(frame)
        x_offset = max((frame.shape[1] - new_w) // 2, 0)
        hand[:, x_offset : x_offset + min(new_w, frame.shape[1])] = resized[
            :, : frame.shape[1] - x_offset
        ]

        display_combined(frame, mask, hand)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def train_model():
    labels_dict = {"one": 0, "two": 1, "three": 2, "four": 3, "five": 4}
    data_dir = "data/raw"
    X = []
    y = []

    for gesture in os.listdir(data_dir):
        gesture_path = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue

        label = labels_dict[gesture]

        for file in os.listdir(gesture_path):
            if not file.endswith(".png"):
                continue

            img_path = os.path.join(gesture_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            hand_img = preprocess_hand(img)
            features = compute_hog_features(hand_img)

            X.append(features)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", len(list(filter(lambda x: x == 2, y))))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(X_scaled.shape)
    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X_scaled)

    print("X_pca shape:", X_pca.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.3, random_state=42
    )

    svm = LinearSVC(max_iter=5000)
    svm.fit(X_train, y_train)

    scores = cross_val_score(svm, X_pca, y, cv=5)
    print(scores, scores.mean())

    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"scaler": scaler, "pca": pca, "svm": svm}, "svm_hog_pipeline.pkl")


# train_model()
video()
