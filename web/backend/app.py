from flask import Flask, request, jsonify
import numpy as np
from model import SoftmaxRegression
import pickle
import base64
from PIL import Image
from flask_cors import CORS
import io
import json

app = Flask(__name__)
CORS(app) 

# Load trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# -----------------------------
# Convert base64 → 28x28 features
# -----------------------------

def convert_image_to_features(base64_str):
    # Decode base64 → bytes
    img_bytes = base64.b64decode(base64_str.split(",")[1])
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Resize xuống 14×14 = 196 pixel
    img = img.resize((14, 14))

    arr = np.array(img) / 255.0
    arr = arr.reshape(-1)  # → (196,)

    # Add bias feature → thành 197
    arr = np.append(arr, 1.0)

    return arr

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
        features = convert_image_to_features(data["image"])

        # Model prediction
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0]

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
