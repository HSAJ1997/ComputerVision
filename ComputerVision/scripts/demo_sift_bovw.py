import random
import cv2
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 500

def get_random_image():
    subset_root = PROJECT_ROOT / "subset/train_mini"
    all_images = list(subset_root.rglob("*.jpg"))
    img_path = random.choice(all_images)
    print(f"Randomly selected image:\n{img_path}\n")

    return img_path

def show_raw_image(image_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.title("Raw Image")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

    return img

def show_sift_keypoints(image_path):
    img = cv2.imread(str(image_path))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    print(f"Image: {image_path.name}")
    print(f"Number of keypoints detected: {len(keypoints)}")
    print(f"Descriptors shape: {descriptors.shape}")

    img_kp = cv2.drawKeypoints(gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_rgb = cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.title("SIFT Keypoints")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()

def show_bovw_encoding(image_path, kmeans):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    word_ids = kmeans.predict(descriptors)
    hist, _ = np.histogram(word_ids, bins=np.arange(VOCAB_SIZE + 1))
    hist = hist.astype(np.float32)
    hist /= np.linalg.norm(hist) if np.linalg.norm(hist) > 0 else 1.0 

    print("BoVW Histogram shape:", hist.shape)
    print("BoVW Histogram sum", np.sum(hist))

    print("First 20 Histogram bins:", hist[:20])

    plt.figure(figsize=(10, 4))
    plt.title("BoVW Histogram (first 100 bins)")
    plt.plot(hist[:100])
    plt.xlabel("Visual Word Index")
    plt.ylabel("Normalized Frequency")
    plt.show()

    return hist

def main():
    img_path = get_random_image()
    img = show_raw_image(img_path)
    if img is None:
        return

    show_sift_keypoints(img_path)
    vocab_path = PROJECT_ROOT / "bovw_kmeans.pkl"

    print("\nLoading vocabulary from:", vocab_path.name)
    kmeans = joblib.load(vocab_path)
    print("Vocabulary loaded. Shape:", kmeans.cluster_centers_.shape)

    hist = show_bovw_encoding(img_path, kmeans)

if __name__ == "__main__":
    main()