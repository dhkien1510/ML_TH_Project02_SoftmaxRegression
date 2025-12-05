# HƯỚNG DẪN SỬ DỤNG SOFTMAX REGRESSION MODEL
---

## MỤC LỤC

1. [Tổng quan](#1-tổng-quan)
2. [Cài đặt và Import](#2-cài-đặt-và-import)
3. [Khởi tạo Model](#3-khởi-tạo-model)
4. [Training Model](#4-training-model)
5. [Prediction và Evaluation](#5-prediction-và-evaluation)
6. [Thiết kế Feature Vector](#6-thiết-kế-feature-vector)
7. [Tích hợp vào Web Application](#7-tích-hợp-vào-web-application)
8. [Lưu và Load Model](#8-lưu-và-load-model)
9. [Best Practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. TỔNG QUAN

### 1.1. Model là gì?

`SoftmaxRegression` là một model phân loại đa lớp (multi-class classification) được implement hoàn toàn bằng NumPy, tuân thủ theo các công thức toán học chuẩn trong Machine Learning.

### 1.2. Tính năng chính

- **Softmax activation** cho multi-class classification
- **Cross-Entropy loss** (sparse format)
- **L2 Regularization** để tránh overfitting
- **Mini-batch Stochastic Gradient Descent** (SGD)
- **Z-score Normalization** tự động
- **Xavier/Glorot Weight Initialization**
- **Numerical Gradient Checking** để verify tính đúng đắn

---

## 2. CÀI ĐẶT VÀ IMPORT

### 2.1. Dependencies

```python
import numpy as np
```

**Lưu ý:** Model chỉ cần NumPy, không phụ thuộc vào thư viện ML nào khác.

### 2.2. Import Model

```python
from SoftmaxRegression import SoftmaxRegression
```

---

## 3. KHỞI TẠO MODEL

### 3.1. Constructor

```python
model = SoftmaxRegression(
    learning_rate=0.1,      # Tốc độ học
    epochs=100,             # Số epoch training
    batch_size=128,         # Kích thước mini-batch
    reg=1e-4,               # Hệ số regularization (lambda)
    normalize=True,         # Bật/tắt normalization
    random_state=42,        # Seed cho reproducibility
    verbose=True            # Hiển thị tiến trình training
)
```

### 3.2. Các tham số chi tiết

| Tham số | Kiểu | Mặc định | Mô tả |
|---------|------|----------|-------|
| `learning_rate` | float | 0.1 | Tốc độ học (α). Giá trị cao → học nhanh nhưng không ổn định |
| `epochs` | int | 100 | Số lần duyệt qua toàn bộ dataset |
| `batch_size` | int | 128 | Số samples trong mỗi mini-batch. Nhỏ → nhiễu cao, lớn → ổn định hơn |
| `reg` | float | 1e-4 | Hệ số L2 regularization (λ). Càng lớn → model càng đơn giản |
| `normalize` | bool | True | **Quan trọng!** Luôn để `True` để chuẩn hóa dữ liệu |
| `random_state` | int/None | None | Seed cho NumPy để kết quả reproducible |
| `verbose` | bool | True | In loss mỗi 10 epochs |

### 3.3. Ví dụ khởi tạo

```python
# Model đơn giản cho testing
model_simple = SoftmaxRegression(
    learning_rate=0.01,
    epochs=50,
    verbose=False
)

# Model cho production với hyperparameters tối ưu
model_prod = SoftmaxRegression(
    learning_rate=0.05,
    epochs=200,
    batch_size=256,
    reg=5e-4,
    normalize=True,
    random_state=42,
    verbose=True
)
```

---

## 4. TRAINING MODEL

### 4.1. Chuẩn bị dữ liệu

**Yêu cầu đầu vào:**

```python
# X: Feature matrix
# - Shape: (n_samples, n_features)
# - Type: numpy array hoặc list
# - Ví dụ: (1000, 784) cho MNIST

# y: Labels
# - Shape: (n_samples,)
# - Type: numpy array hoặc list với giá trị integer
# - Giá trị: 0, 1, 2, ..., n_classes-1
# - Ví dụ: [0, 1, 2, 0, 1, ...] cho 3 classes
```

**Ví dụ:**

```python
import numpy as np

# Ví dụ 1: Dữ liệu ngẫu nhiên
X_train = np.random.randn(1000, 784)  # 1000 samples, 784 features
y_train = np.random.randint(0, 10, size=1000)  # 10 classes (0-9)

# Ví dụ 2: Dữ liệu thực tế
# X_train có thể là feature vector từ hình ảnh, text, etc.
# y_train là nhãn tương ứng
```

### 4.2. Train Model

```python
# Train model
model.fit(X_train, y_train)
```

**Output trong quá trình training:**

```
Epoch 1/100 | Loss: 2.301234
Epoch 10/100 | Loss: 0.876543
Epoch 20/100 | Loss: 0.654321
...
Epoch 100/100 | Loss: 0.234567
```

### 4.3. Quy trình bên trong `.fit()`

1. **Normalization** (nếu `normalize=True`):
   - Tính mean và std của X_train
   - Transform: `X = (X - mean) / std`
   - Lưu mean/std để dùng cho prediction

2. **Weight Initialization**:
   - Xavier initialization: `W ~ N(0, 1/sqrt(n_features))`
   - Bias: `b = 0`

3. **Mini-batch SGD**:
   - Mỗi epoch: shuffle data → chia thành batches
   - Mỗi batch: tính loss & gradients → update weights

---

## 5. PREDICTION VÀ EVALUATION

### 5.1. Predict Class Labels

```python
# Dự đoán class cho dữ liệu mới
y_pred = model.predict(X_test)

# Output: numpy array shape (n_samples,)
# Ví dụ: array([0, 2, 1, 0, 3, ...])
```

### 5.2. Predict Probabilities

```python
# Dự đoán xác suất cho từng class
probs = model.predict_proba(X_test)

# Output: numpy array shape (n_samples, n_classes)
# Mỗi hàng là phân phối xác suất, tổng = 1
# Ví dụ với 3 classes:
# array([[0.8, 0.15, 0.05],   # Sample 1: 80% class 0, 15% class 1, 5% class 2
#        [0.1, 0.7, 0.2],     # Sample 2: 10% class 0, 70% class 1, 20% class 2
#        ...])
```

**Ứng dụng probabilities:**

```python
# Lấy class với xác suất cao nhất
predicted_class = np.argmax(probs, axis=1)

# Lấy confidence (xác suất cao nhất)
confidence = np.max(probs, axis=1)

# Lọc predictions với confidence > 0.9
high_conf_indices = confidence > 0.9
reliable_predictions = y_pred[high_conf_indices]
```

### 5.3. Evaluate Accuracy

```python
# Tính accuracy trên test set
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")  # Output: Test Accuracy: 0.9234
```

### 5.4. Ví dụ đầy đủ

```python
# Train
model = SoftmaxRegression(learning_rate=0.1, epochs=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predict
predictions = model.predict(X_test[:5])
probabilities = model.predict_proba(X_test[:5])

for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Sample {i}: Predicted class = {pred}, Probabilities = {prob}")
```

---

## 6. THIẾT KẾ FEATURE VECTOR

### 6.1. Yêu cầu cho Feature Vector

**Quan trọng!** Feature vector phải đảm bảo:

1. **Shape**: `(n_samples, n_features)` hoặc `(n_features,)` cho 1 sample
2. **Type**: NumPy array hoặc list (sẽ tự động convert)
3. **Values**: Số thực (float), không có NaN hoặc Inf

### 6.2. Ví dụ Feature Extraction

#### 6.2.1. Từ Hình ảnh

```python
from PIL import Image
import numpy as np

def extract_image_features(image_path):
    """
    Extract features từ hình ảnh
    
    Args:
        image_path: đường dẫn đến file ảnh
    
    Returns:
        feature_vector: numpy array shape (n_features,)
    """
    # Load và resize ảnh
    img = Image.open(image_path).convert('L')  # Grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    
    # Convert to numpy array và flatten
    pixels = np.array(img).flatten()  # Shape: (784,)
    
    # Normalize pixel values to [0, 1]
    features = pixels.astype(float) / 255.0
    
    return features

# Sử dụng
img_features = extract_image_features('digit.png')
print(img_features.shape)  # (784,)

# Predict cho 1 ảnh
prediction = model.predict([img_features])  # Wrap in list
print(f"Predicted class: {prediction[0]}")

# Predict cho nhiều ảnh
images = ['img1.png', 'img2.png', 'img3.png']
features_batch = np.array([extract_image_features(img) for img in images])
predictions = model.predict(features_batch)
```

#### 6.2.2. Từ Text

```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text_features(texts, vectorizer=None, train=False):
    """
    Extract TF-IDF features từ text
    
    Args:
        texts: list of strings
        vectorizer: TfidfVectorizer object (None nếu train=True)
        train: True khi training, False khi prediction
    
    Returns:
        features: numpy array shape (n_samples, n_features)
        vectorizer: fitted vectorizer (nếu train=True)
    """
    if train:
        vectorizer = TfidfVectorizer(max_features=1000)
        features = vectorizer.fit_transform(texts).toarray()
        return features, vectorizer
    else:
        features = vectorizer.transform(texts).toarray()
        return features

# Training
texts_train = ["This is good", "This is bad", ...]
X_train, vectorizer = extract_text_features(texts_train, train=True)
model.fit(X_train, y_train)

# Prediction
texts_test = ["New text to classify"]
X_test = extract_text_features(texts_test, vectorizer=vectorizer, train=False)
prediction = model.predict(X_test)
```

#### 6.2.3. Feature Vector Tổng quát

```python
def create_feature_vector(data_point):
    """
    Template cho feature extraction function
    
    Args:
        data_point: dữ liệu đầu vào (ảnh, text, audio, etc.)
    
    Returns:
        features: numpy array shape (n_features,)
    """
    # Bước 1: Extract raw features
    raw_features = extract_raw_features(data_point)
    
    # Bước 2: Preprocessing (optional)
    # - Normalization: đã được model tự động xử lý
    # - Feature scaling: nên làm nếu các features có scale khác nhau
    processed_features = preprocess_features(raw_features)
    
    # Bước 3: Ensure correct format
    features = np.array(processed_features, dtype=float)
    
    # Bước 4: Validate
    assert not np.any(np.isnan(features)), "Features contain NaN"
    assert not np.any(np.isinf(features)), "Features contain Inf"
    assert features.ndim == 1, "Features must be 1D array"
    
    return features

# Sử dụng
feature = create_feature_vector(my_data)
prediction = model.predict([feature])
```

### 6.3. Best Practices cho Feature Engineering

```python
# O: Consistent feature dimension
# Tất cả feature vectors phải có cùng số chiều
X_train = np.random.randn(1000, 784)  # 784 features
X_test = np.random.randn(200, 784)    # Cùng 784 features

#DON'T: Inconsistent dimensions
# X_test = np.random.randn(200, 512)  # Sai! Khác số features

#DO: Handle missing values
def handle_missing(features):
    # Replace NaN with mean
    nan_mask = np.isnan(features)
    features[nan_mask] = np.nanmean(features)
    return features

# ✅ DO: Feature scaling (optional, nhưng recommended)
# Model đã tự normalize, nhưng nếu features có scale rất khác nhau:
def scale_features(features):
    # Min-max scaling to [0, 1]
    min_val = features.min()
    max_val = features.max()
    if max_val > min_val:
        return (features - min_val) / (max_val - min_val)
    return features
```

---

## 7. TÍCH HỢP VÀO WEB APPLICATION

### 7.1. Architecture Overview

```
User Input (Web) → Feature Extraction → Model Prediction → Response (JSON)
```

### 7.2. Flask Application Example

```python
from flask import Flask, request, jsonify
import numpy as np
from SoftmaxRegression import SoftmaxRegression
import pickle

app = Flask(__name__)

# Load trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint cho prediction
    
    Request JSON:
    {
        "features": [0.1, 0.2, ..., 0.9]  # List of feature values
    }
    
    Response JSON:
    {
        "predicted_class": 3,
        "probabilities": [0.05, 0.1, 0.15, 0.7],
        "confidence": 0.7
    }
    """
    try:
        # Parse input
        data = request.get_json()
        features = np.array(data['features'], dtype=float)
        
        # Validate input
        if features.ndim == 1:
            features = features.reshape(1, -1)  # Shape: (1, n_features)
        
        # Predict
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        # Response
        return jsonify({
            'success': True,
            'predicted_class': int(prediction),
            'probabilities': probabilities.tolist(),
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    API endpoint cho batch prediction
    
    Request JSON:
    {
        "features_list": [
            [0.1, 0.2, ..., 0.9],
            [0.2, 0.3, ..., 0.8],
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        features_list = np.array(data['features_list'], dtype=float)
        
        predictions = model.predict(features_list)
        probabilities = model.predict_proba(features_list)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'index': i,
                'predicted_class': int(pred),
                'probabilities': prob.tolist(),
                'confidence': float(np.max(prob))
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 7.3. FastAPI Example (Modern Alternative)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List
import pickle

app = FastAPI()

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    success: bool
    predicted_class: int
    probabilities: List[float]
    confidence: float

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict endpoint với automatic validation"""
    try:
        features = np.array(request.features, dtype=float).reshape(1, -1)
        
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        return PredictionResponse(
            success=True,
            predicted_class=int(prediction),
            probabilities=probabilities.tolist(),
            confidence=float(np.max(probabilities))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### 7.4. Client-side JavaScript Example

```javascript
// Function để gọi API prediction
async function predictClass(features) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: features
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            console.log('Predicted class:', result.predicted_class);
            console.log('Confidence:', result.confidence);
            console.log('Probabilities:', result.probabilities);
            return result;
        } else {
            console.error('Prediction failed:', result.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Ví dụ sử dụng
const myFeatures = [0.1, 0.2, 0.3, ..., 0.9];  // 784 features
predictClass(myFeatures);
```

### 7.5. Error Handling Best Practices

```python
def safe_predict(model, features):
    """
    Wrapper function với comprehensive error handling
    """
    try:
        # Validate input type
        if not isinstance(features, (np.ndarray, list)):
            raise ValueError("Features must be numpy array or list")
        
        # Convert to numpy array
        features = np.array(features, dtype=float)
        
        # Validate shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        elif features.ndim != 2:
            raise ValueError(f"Invalid shape: {features.shape}")
        
        # Check for NaN/Inf
        if np.any(np.isnan(features)):
            raise ValueError("Features contain NaN values")
        if np.any(np.isinf(features)):
            raise ValueError("Features contain Inf values")
        
        # Check feature dimension
        expected_features = model.W.shape[0]
        if features.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {features.shape[1]}"
            )
        
        # Predict
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        
        return {
            'success': True,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
```

---

## 8. LƯU VÀ LOAD MODEL

### 8.1. Sử dụng Pickle (Recommended)

```python
import pickle

# === LƯU MODEL ===
# Sau khi train xong
model.fit(X_train, y_train)

# Lưu model
with open('softmax_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!")

# === LOAD MODEL ===
# Load model đã train
with open('softmax_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Sử dụng ngay
predictions = loaded_model.predict(X_test)
```

### 8.2. Lưu riêng Weights và Parameters

```python
import json

# === LƯU ===
def save_model_weights(model, filepath):
    """Lưu weights và parameters"""
    model_data = {
        'W': model.W.tolist(),
        'b': model.b.tolist(),
        'scaler_mean': model.scaler_mean.tolist() if model.scaler_mean is not None else None,
        'scaler_std': model.scaler_std.tolist() if model.scaler_std is not None else None,
        'hyperparameters': {
            'learning_rate': model.learning_rate,
            'reg': model.reg,
            'normalize': model.normalize
        }
    }
    
    # Lưu weights as numpy file
    np.save(filepath + '_weights.npy', model_data)
    
    # Hoặc lưu as JSON (cho portability)
    with open(filepath + '_weights.json', 'w') as f:
        json.dump(model_data, f)

# === LOAD ===
def load_model_weights(filepath, model_class=SoftmaxRegression):
    """Load weights và tạo model"""
    # Load từ numpy file
    model_data = np.load(filepath + '_weights.npy', allow_pickle=True).item()
    
    # Tạo model mới
    model = model_class(
        learning_rate=model_data['hyperparameters']['learning_rate'],
        reg=model_data['hyperparameters']['reg'],
        normalize=model_data['hyperparameters']['normalize']
    )
    
    # Set weights
    model.W = np.array(model_data['W'])
    model.b = np.array(model_data['b'])
    
    if model_data['scaler_mean'] is not None:
        model.scaler_mean = np.array(model_data['scaler_mean'])
        model.scaler_std = np.array(model_data['scaler_std'])
    
    return model

# Sử dụng
save_model_weights(model, 'my_model')
loaded_model = load_model_weights('my_model')
```

### 8.3. Version Control cho Models

```python
import datetime
import os

def save_model_with_version(model, model_name, metrics=None):
    """
    Lưu model với timestamp và metrics
    
    Args:
        model: trained model
        model_name: tên model (vd: 'mnist_classifier')
        metrics: dict chứa metrics (vd: {'accuracy': 0.95})
    """
    # Tạo timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Tạo filename
    filename = f'models/{model_name}_{timestamp}.pkl'
    
    # Lưu model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Lưu metadata
    metadata = {
        'timestamp': timestamp,
        'model_name': model_name,
        'metrics': metrics or {},
        'hyperparameters': {
            'learning_rate': model.learning_rate,
            'epochs': model.epochs,
            'batch_size': model.batch_size,
            'reg': model.reg
        }
    }
    
    metadata_file = f'models/{model_name}_{timestamp}_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved: {filename}")
    print(f"Metadata saved: {metadata_file}")
    return filename

# Sử dụng
metrics = {
    'train_accuracy': 0.98,
    'test_accuracy': 0.95,
    'loss': 0.234
}
save_model_with_version(model, 'mnist_softmax', metrics)
```

---

## 9. BEST PRACTICES

### 9.1. Hyperparameter Tuning

```python
from sklearn.model_selection import train_test_split

# Split data
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Grid search
learning_rates = [0.01, 0.05, 0.1, 0.5]
reg_values = [1e-5, 1e-4, 1e-3]
batch_sizes = [64, 128, 256]

best_accuracy = 0
best_params = {}

for lr in learning_rates:
    for reg in reg_values:
        for bs in batch_sizes:
            # Train model
            model = SoftmaxRegression(
                learning_rate=lr,
                epochs=100,
                batch_size=bs,
                reg=reg,
                random_state=42,
                verbose=False
            )
            model.fit(X_train_full, y_train_full)
            
            # Evaluate
            val_acc = model.score(X_val, y_val)
            
            # Track best
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = {
                    'learning_rate': lr,
                    'reg': reg,
                    'batch_size': bs
                }

print(f"Best params: {best_params}")
print(f"Best validation accuracy: {best_accuracy:.4f}")

# Train final model với best params
final_model = SoftmaxRegression(**best_params, epochs=200, random_state=42)
final_model.fit(X_train, y_train)
```

### 9.2. Monitoring Training

```python
import matplotlib.pyplot as plt

# Train model
model = SoftmaxRegression(epochs=200, verbose=True)
model.fit(X_train, y_train)

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(model.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

# Check for overfitting
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Gap: {train_acc - test_acc:.4f}")

if train_acc - test_acc > 0.1:
    print("Warning: Possible overfitting! Consider:")
    print("  - Increase regularization (reg parameter)")
    print("  - Reduce model complexity")
    print("  - Add more training data")
```

### 9.3. Cross-Validation

```python
from sklearn.model_selection import KFold

def cross_validate(X, y, n_splits=5):
    """K-Fold Cross-Validation"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        model = SoftmaxRegression(
            learning_rate=0.1,
            epochs=100,
            random_state=42,
            verbose=False
        )
        model.fit(X_train_cv, y_train_cv)
        
        acc = model.score(X_val_cv, y_val_cv)
        accuracies.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.4f}")
    
    print(f"\nMean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
    return accuracies

# Sử dụng
cross_validate(X_train, y_train, n_splits=5)
```

### 9.4. Gradient Checking (Verify Implementation)

```python
# Kiểm tra implementation có đúng không
model = SoftmaxRegression()
is_correct = model.gradient_check(X_train, y_train)

if is_correct:
    print("Gradient implementation is correct!")
else:
    print("Gradient implementation has errors!")
```

---

## 10. TROUBLESHOOTING

### 10.1. Common Issues

#### Issue 1: Loss không giảm

```python
# Nguyên nhân có thể:
# - Learning rate quá cao hoặc quá thấp
# - Regularization quá mạnh
# - Data không được normalize

# Giải pháp:
model = SoftmaxRegression(
    learning_rate=0.01,  # Giảm learning rate
    reg=1e-5,            # Giảm regularization
    normalize=True,      # Đảm bảo normalize=True
    epochs=200           # Tăng epochs
)
```

#### Issue 2: Accuracy quá thấp

```python
# Kiểm tra:
# 1. Feature quality
print(f"Feature shape: {X_train.shape}")
print(f"Feature range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"NaN count: {np.isnan(X_train).sum()}")

# 2. Label distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# 3. Model capacity
# Ensure model có đủ capacity (số features đủ lớn)
```

#### Issue 3: Overfitting

```python
# Giải pháp:
model = SoftmaxRegression(
    reg=1e-3,           # Tăng regularization
    epochs=100          # Giảm epochs
)

# Hoặc: Thu thập thêm data
```

#### Issue 4: Dimension Mismatch

```python
# Error: "shapes (100, 512) and (784, 10) not aligned"

# Nguyên nhân: X_test có số features khác X_train
print(f"Train features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")

# Giải pháp: Đảm bảo feature extraction giống nhau
```

### 10.2. Debugging Checklist

```python
def debug_model(model, X_train, y_train, X_test, y_test):
    """Comprehensive debugging"""
    print("=== DEBUGGING MODEL ===\n")
    
    # 1. Data checks
    print("1. DATA CHECKS")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")
    print(f"   Train labels: {np.unique(y_train)}")
    print(f"   Test labels: {np.unique(y_test)}")
    print(f"   NaN in train: {np.isnan(X_train).sum()}")
    print(f"   NaN in test: {np.isnan(X_test).sum()}\n")
    
    # 2. Model parameters
    print("2. MODEL PARAMETERS")
    print(f"   Learning rate: {model.learning_rate}")
    print(f"   Epochs: {model.epochs}")
    print(f"   Batch size: {model.batch_size}")
    print(f"   Regularization: {model.reg}")
    print(f"   Normalize: {model.normalize}\n")
    
    # 3. Training
    print("3. TRAINING")
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    print("\n4. EVALUATION")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    print(f"   Overfitting gap: {train_acc - test_acc:.4f}\n")
    
    # 5. Predictions
    print("5. SAMPLE PREDICTIONS")
    probs = model.predict_proba(X_test[:5])
    preds = model.predict(X_test[:5])
    for i in range(5):
        print(f"   Sample {i}: True={y_test[i]}, Pred={preds[i]}, "
              f"Conf={probs[i].max():.3f}")

# Sử dụng
debug_model(model, X_train, y_train, X_test, y_test)
```

---

## APPENDIX

### A. Complete Working Example

```python
"""
Complete example: Train và sử dụng Softmax Regression
"""
import numpy as np
from SoftmaxRegression import SoftmaxRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pickle

# 1. Load data
print("Loading data...")
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model
print("\nCreating model...")
model = SoftmaxRegression(
    learning_rate=0.5,
    epochs=200,
    batch_size=128,
    reg=1e-4,
    normalize=True,
    random_state=42,
    verbose=True
)

# 4. Train
print("\nTraining model...")
model.fit(X_train, y_train)

# 5. Evaluate
print("\nEvaluating...")
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\nFinal Results:")
print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 6. Save model
print("\nSaving model...")
with open('digit_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

# 7. Test predictions
print("\nTesting predictions...")
sample = X_test[0:1]
pred = model.predict(sample)[0]
prob = model.predict_proba(sample)[0]
print(f"True label: {y_test[0]}")
print(f"Predicted: {pred}")
print(f"Probabilities: {prob}")
print(f"Confidence: {prob.max():.4f}")

print("\nComplete!")
```

### B. Class Attributes Reference

| Attribute | Type | Description |
|-----------|------|-------------|
| `model.W` | numpy array | Weight matrix, shape (n_features, n_classes) |
| `model.b` | numpy array | Bias vector, shape (1, n_classes) |
| `model.scaler_mean` | numpy array | Mean for normalization, shape (n_features,) |
| `model.scaler_std` | numpy array | Std for normalization, shape (n_features,) |
| `model.history` | dict | Training history, keys: ['loss'] |
| `model.n_samples` | int | Number of training samples |

### C. Method Reference

| Method | Description | Returns |
|--------|-------------|---------|
| `fit(X, y)` | Train model | self |
| `predict(X)` | Predict class labels | numpy array (n_samples,) |
| `predict_proba(X)` | Predict probabilities | numpy array (n_samples, n_classes) |
| `score(X, y)` | Calculate accuracy | float |
| `gradient_check(X, y)` | Verify gradients | bool |

---

## SUPPORT

Nếu gặp vấn đề, hãy check:

1. **Data format**: X shape (n_samples, n_features), y shape (n_samples,)
2. **Feature consistency**: Train và test phải có cùng số features
3. **Normalization**: Luôn để `normalize=True`
4. **Label format**: Labels phải là integers từ 0 đến n_classes-1



