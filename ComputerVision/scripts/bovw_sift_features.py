from pathlib import Path
import csv
import joblib
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VOCAB_SIZE = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def load_split(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)
    return rows


def extract_sift_descriptors(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return np.empty((0, 128), dtype=np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)

    return descriptors.astype(np.float32)

def build_vocabulary(train_rows):
    all_desc = []

    print("Collecting SIFT descriptors from training images")

    for row in train_rows:
        image_path = PROJECT_ROOT / row["image_path"]
        desc = extract_sift_descriptors(image_path)

        if desc.shape[0] > 0:
            all_desc.append(desc)

        all_desc = np.vstack(all_desc)
        print(f"Total descriptors collected: {all_desc.shape[0]}")

        print("Running k-means...")
        kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=1000, random_state=RANDOM_SEED)
        kmeans.fit(all_desc)

        joblib.dump(kmeans, PROJECT_ROOT / "bovw_kmeans.pkl")
        print("Vocabulary saved to bovw_kmeans.pkl")

        return kmeans

def encode_image(image_path, kmeans):
    desc = extract_sift_descriptors(image_path)
    if desc.shape[0] == 0:
        return np.zeros(VOCAB_SIZE, dtype=np.float32)

    word_ids = kmeans.predict(desc)

    hist, _ = np.histogram(word_ids, bins=np.arange(VOCAB_SIZE + 1))
    hist = hist.astype(np.float32)

    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm

    return hist

def encode_split(rows, kmeans, name):
    features = []
    labels = []

    print(f"Encoding {name} split...")

    for row in rows:
        image_path = PROJECT_ROOT / row["image_path"]
        class_index = int(row["class_index"])

        hist = encode_image(image_path, kmeans)

        features.append(hist)
        labels.append(class_index)

    features = np.stack(features)
    labels = np.array(labels)

    np.save(PROJECT_ROOT / f"bovw_{name}_features.npy", features)
    np.save(PROJECT_ROOT / f"bovw_{name}_labels.npy", labels)

    print(f"Saved {name} bovw features and labels")


def main():
    train_rows = load_split(PROJECT_ROOT / "splits/train.csv")
    val_rows = load_split(PROJECT_ROOT / "splits/validation.csv")
    test_rows = load_split(PROJECT_ROOT / "splits/test.csv")

    kmeans = build_vocabulary(train_rows)

    encode_split(train_rows, kmeans, "train")
    encode_split(val_rows, kmeans, "validation")
    encode_split(test_rows, kmeans, "test")

    print("BoVW feature extraction complete.")

if __name__ == "__main__":
    main()