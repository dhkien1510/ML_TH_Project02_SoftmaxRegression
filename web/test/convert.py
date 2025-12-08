import numpy as np
from PIL import Image

def image_to_features(image_path, resize_to=(28, 28)):
    # Load image
    img = Image.open(image_path).convert("L")  # convert to grayscale

    # Resize
    img = img.resize(resize_to)

    # Convert to numpy array
    arr = np.array(img, dtype=float)

    # Normalize 0–1
    arr = arr / 255.0

    # Flatten to 1D vector
    features = arr.flatten()

    return features.tolist()

if __name__ == "__main__":
    image_path = "sample.jpg"   # ✔️ đổi tên file ảnh vào đây
    features = image_to_features(image_path)

    # Print as JSON for .http file
    print("Copy this JSON into your .http file:")
    print("\n{\n    \"features\": [")
    
    # Pretty print 784 elements
    for i, val in enumerate(features):
        end = "," if i < len(features) - 1 else ""
        print(f"        {val}{end}")
    
    print("    ]\n}")
