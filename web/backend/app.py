from flask import Flask, request, jsonify
import numpy as np
from model import SoftmaxRegression
import pickle
import base64
from PIL import Image, ImageOps
from flask_cors import CORS
import io
import json

app = Flask(__name__)
CORS(app) 

# Load trained model
with open('design_2_block_avg_2x2.pkl', 'rb') as f:
    model = pickle.load(f)
    
# -----------------------------
# Convert base64 → 28x28 features
# -----------------------------



def convert_image_to_features(base64_str):
    # 1. Decode base64 -> Image
    img_data = base64.b64decode(base64_str.split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert('L') 

    # --- BƯỚC 1: XỬ LÝ MÀU SẮC & TƯƠNG PHẢN ---
    if np.array(img)[0, 0] > 128:
        img = ImageOps.invert(img)
    
    img = img.point(lambda x: 255 if x > 100 else 0, mode='1').convert('L')

    img_arr = np.array(img)

    coords = np.argwhere(img_arr > 0)
    
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
   
        cropped = img.crop((x_min, y_min, x_max + 1, y_max + 1))
        
       
        w, h = cropped.size
        scale = 20.0 / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        new_img = Image.new('L', (28, 28), 0)
        
      
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        new_img.paste(resized_digit, (paste_x, paste_y))
        
        img = new_img
    else:
        # Nếu ảnh đen xì không có gì thì giữ nguyên resize thường
        img = img.resize((28, 28))

   
    # --- BƯỚC 3: FEATURE ENGINEERING (Giữ nguyên logic cũ của bạn) ---
    X_img = np.array(img) / 255.0 # Normalize về [0, 1]

    block_size = 2
    n_blocks = 28 // block_size
    features = []

    for i in range(n_blocks):
        for j in range(n_blocks):
            block = X_img[
                i*block_size : (i+1)*block_size,
                j*block_size : (j+1)*block_size
            ]
            features.append(block.mean())

    arr = np.array(features)

    # Thêm Bias
    arr = np.insert(arr, 0, 1.0) 
    
    return arr.reshape(1, -1)

# -----------------------------
# Predict API
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """
    FE gửi:
    {
        "image": "data:image/png;base64,iVBORw0...."
    }
    """

    try:
        data = request.get_json()

        if "image" not in data:
            return jsonify({'success': False, 'error': 'Missing image field'}), 400

        # Convert base64 → 784 features
        # Trong hàm predict()
        features = convert_image_to_features(data["image"]) # Trả về shape (1, 197)

        # Không cần đóng thêm ngoặc vuông [] bên ngoài nữa vì nó đã là 2D rồi
        pred = model.predict(features)[0] 
        prob = model.predict_proba(features)[0]

        print(json.dumps({
            "success": True,
            "predicted_class": int(pred),
            "probabilities": prob.tolist(),
            "confidence": float(np.max(prob))
        }, indent=4))
        
        return jsonify({
            "success": True,
            "predicted_class": int(pred),
            "probabilities": prob.tolist(),
            "confidence": float(np.max(prob))
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# -----------------------------
# Batch API (optional)
# -----------------------------
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        images = data["images"]   # list base64

        features_list = [convert_image_to_features(img) for img in images]

        preds = model.predict(features_list)
        probs = model.predict_proba(features_list)

        results = []
        for i, (p, pr) in enumerate(zip(preds, probs)):
            results.append({
                "index": i,
                "predicted_class": int(p),
                "probabilities": pr.tolist(),
                "confidence": float(np.max(pr))
            })

        return jsonify({"success": True, "results": results})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
