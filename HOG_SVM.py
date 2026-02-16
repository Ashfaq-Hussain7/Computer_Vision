import os
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# PARAMETERS
WIN_W, WIN_H = 64, 128
NEG_PER_IMAGE = 2

# PATHS
BASE_DIR = "INRIA"
OUT_DIR = "processed_INRIA"


# CREATE OUTPUT STRUCTURE
for split in ["train", "test"]:
    os.makedirs(f"{OUT_DIR}/{split}/pos", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/{split}/neg", exist_ok=True)


# FUNCTION: XML → POS / NEG
def prepare_data(split):
    img_dir = f"{BASE_DIR}/{split.capitalize()}/JPEGImages"
    ann_dir = f"{BASE_DIR}/{split.capitalize()}/Annotations"
    pos_dir = f"{OUT_DIR}/{split}/pos"
    neg_dir = f"{OUT_DIR}/{split}/neg"

    pos_id, neg_id = 0, 0

    for ann_file in os.listdir(ann_dir):
        tree = ET.parse(os.path.join(ann_dir, ann_file))
        root = tree.getroot()

        img_name = root.find("filename").text
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape
        person_boxes = []

        # POS samples
        for obj in root.findall("object"):
            if obj.find("name").text.lower() == "person":
                bnd = obj.find("bndbox")
                xmin = int(bnd.find("xmin").text)
                ymin = int(bnd.find("ymin").text)
                xmax = int(bnd.find("xmax").text)
                ymax = int(bnd.find("ymax").text)

                person_boxes.append((xmin, ymin, xmax, ymax))
                crop = img[ymin:ymax, xmin:xmax]

                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (WIN_W, WIN_H))
                cv2.imwrite(f"{pos_dir}/{pos_id}.jpg", crop)
                pos_id += 1

        # NEG samples
        for _ in range(NEG_PER_IMAGE):
            for _ in range(20):
                x = random.randint(0, w - WIN_W)
                y = random.randint(0, h - WIN_H)

                overlap = False
                for (xmin, ymin, xmax, ymax) in person_boxes:
                    if x < xmax and x + WIN_W > xmin and y < ymax and y + WIN_H > ymin:
                        overlap = True
                        break

                if not overlap:
                    neg_crop = img[y:y+WIN_H, x:x+WIN_W]
                    cv2.imwrite(f"{neg_dir}/{neg_id}.jpg", neg_crop)
                    neg_id += 1
                    break

    print(f"[{split.upper()}] Pos: {pos_id}, Neg: {neg_id}")

# PREPARE DATA
print("Preparing TRAIN data...")
prepare_data("train")

print("Preparing TEST data...")
prepare_data("test")


# HOG DESCRIPTOR
hog = cv2.HOGDescriptor(
    (64, 128), (16, 16), (8, 8), (8, 8), 9
)

def load_and_extract(folder):
    features = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 128))
            features.append(hog.compute(img).flatten())
    return np.array(features)

# LOAD TRAIN DATA
X_train_pos = load_and_extract(f"{OUT_DIR}/train/pos")
X_train_neg = load_and_extract(f"{OUT_DIR}/train/neg")

X_train = np.vstack((X_train_pos, X_train_neg))
y_train = np.hstack((
    np.ones(len(X_train_pos)),
    np.zeros(len(X_train_neg))
))

# LOAD TEST DATA
X_test_pos = load_and_extract(f"{OUT_DIR}/test/pos")
X_test_neg = load_and_extract(f"{OUT_DIR}/test/neg")

X_test = np.vstack((X_test_pos, X_test_neg))
y_test = np.hstack((
    np.ones(len(X_test_pos)),
    np.zeros(len(X_test_neg))
))

# TRAIN SVM
print("Training Linear SVM...")
svm = LinearSVC(C=0.01, max_iter=10000)
svm.fit(X_train, y_train)

# EVALUATION
print("Evaluating on TEST set...")
y_pred = svm.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
