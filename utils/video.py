import cv2
import numpy as np
import joblib

from constants import (
    CLASSIFY_AREA_THRESHOLD,
    INV_COV,
    MEAN,
    MIN_CONTOUR_AREA,
    NORM_FACTOR,
    SIZE,
)
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


# def preprocess_frame(frame, lower, upper):
#     """Convert frame to YCrCb and apply skin mask with morphology."""
#     ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
#     mask = cv2.inRange(ycrcb, lower, upper)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return mask


def preprocess_frame(frame):
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # https://static.aminer.org/pdf/PDF/000/320/474/a_novel_skin_color_model_in_ycbcr_color_space_and.pdf
    lower = np.array([0, 130, 75])
    upper = np.array([255, 180, 135])
    rect_mask = cv2.inRange(ycrcb, lower, upper)

    _, cr, cb = cv2.split(ycrcb)
    chroma = np.stack((cb, cr), axis=-1)
    diff = chroma - MEAN
    prob = np.exp(-0.5 * np.sum(diff @ INV_COV * diff, axis=-1)) * NORM_FACTOR
    prob_norm = cv2.normalize(prob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, gauss_mask = cv2.threshold(prob_norm, 220, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(rect_mask, gauss_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask, prob_norm


def contour_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def looks_like_eyes(contour, holes):
    x, y, w, h = cv2.boundingRect(contour)

    (x1, y1), (x2, y2) = holes

    # Eyes should be at similar height
    vertical_diff = abs(y1 - y2)
    if vertical_diff > h * 0.15:  # > 15% height difference
        return False

    # Distance between holes should be roughly facial proportion
    horizontal_dist = abs(x1 - x2)
    if horizontal_dist > w * 0.6:
        return False

    # Eyes should be in upper half of blob
    if not (y <= y1 <= y + h * 0.55 and y <= y2 <= y + h * 0.55):
        return False

    return True


def find_valid_contours(mask, min_area=MIN_CONTOUR_AREA):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    valid = []
    if hierarchy is None:
        return valid

    hierarchy = hierarchy[0]

    for i, c in enumerate(contours):

        if cv2.contourArea(c) < min_area:
            continue

        holes = []
        child = hierarchy[i][2]

        while child != -1:
            centroid = contour_centroid(contours[child])
            if centroid is not None:
                holes.append(centroid)
            child = hierarchy[child][0]

        # If two holes detected, check if they look like eyes
        if len(holes) == 2 and not looks_like_eyes(c, holes):
            continue

        valid.append(c)

    return valid


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
        # print(f"Class: {pred_class}, Confidence: {confidence:.4f}")
        final_contours.append((c, confidence, pred_class))

    if not final_contours:
        return None

    # Return contour with highest confidence
    return max(final_contours, key=lambda x: x[1])


def classify_hand(cropped_bin_filled, scaler, pca, model):
    hand_img = preprocess_hand(cropped_bin_filled)
    if hand_img.shape != (SIZE, SIZE):
        return None

    features = compute_hog_features(hand_img)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    pred_class = model.predict(features_pca)[0]
    return pred_class


def draw_prediction(frame, pred_class, history, maxlen):
    labels_dict = {
        0: "one",
        1: "two",
        2: "three",
        3: "four",
        4: "five",
        5: "thumbs_up",
        6: "thumbs_down",
    }
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
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame, mask_bgr, hand))
    cv2.imshow("Frame + Mask + Hand", combined)
