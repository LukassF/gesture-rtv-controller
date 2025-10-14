import cv2
import numpy as np
import joblib

from constants import CLASSIFY_AREA_THRESHOLD, MIN_CONTOUR_AREA, SIZE
from utils.classifier import compute_hog_features, fill_holes, preprocess_hand


def get_cropped_bin(frame, mask, c):
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    y0 = max(y - 1, 0)
    y1 = min(y + h + 1, mask.shape[0])
    x0 = max(x - 1, 0)
    x1 = min(x + w + 1, mask.shape[1])
    cropped_bin = mask[y0:y1, x0:x1]
    cropped_bin[0, :] = 0
    cropped_bin[-1, :] = 0
    cropped_bin[:, 0] = 0
    cropped_bin[:, -1] = 0
    cropped_bin_filled = fill_holes(cropped_bin)
    return cropped_bin_filled


def load_pipeline(path="svm_hog_pipeline.pkl"):
    pipeline = joblib.load(path)
    return pipeline["scaler"], pipeline["pca"], pipeline["svm"]


def preprocess_frame(frame, lower, upper):
    """Convert frame to YCrCb and apply skin mask with morphology."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_valid_contours(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def extract_best_contour(contours, mask, frame, model, scaler, pca):
    final_contours = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < CLASSIFY_AREA_THRESHOLD:
            continue

        cropped_bin_filled = get_cropped_bin(frame, mask, c)
        hand_img = preprocess_hand(cropped_bin_filled)
        if hand_img.shape != (SIZE, SIZE):
            continue

        features = compute_hog_features(hand_img)
        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)

        pred_class = model.predict(features_pca)[0]
        distances = model.decision_function(features_pca)

        confidence = max(distances[0])
        print(f"Class: {pred_class}, Confidence: {confidence:.4f}")
        final_contours.append((c, confidence, pred_class))

    if not final_contours:
        return None

    # Return contour with highest confidence
    return max(final_contours, key=lambda x: x[1])


def classify_hand(cropped_bin_filled, scaler, pca, model):
    """Preprocess and classify a cropped hand region."""
    hand_img = preprocess_hand(cropped_bin_filled)
    if hand_img.shape != (SIZE, SIZE):
        return None

    features = compute_hog_features(hand_img)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    pred_class = model.predict(features_pca)[0]
    return pred_class


def draw_prediction(frame, pred_class, history, maxlen):
    labels_dict = {0: "one", 1: "two", 2: "three", 3: "four", 4: "five"}
    if len(history) == maxlen and history.count(pred_class) > maxlen // 2:
        label_text = labels_dict.get(pred_class, "unknown")
        cv2.putText(
            frame,
            f"Prediction: {label_text}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4,
        )
    else:
        cv2.putText(
            frame, "Detecting...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4
        )


def display_combined(frame, mask, hand):
    """Show frame, mask, and hand side-by-side."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, mask_bgr, hand))
    cv2.imshow("Frame + Mask + Hand", combined)
