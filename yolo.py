from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
import logging
from LLBMA.brain.BMAYOLOManager import YOLO_detect
from LLBMA.resources.BMAassumptions import YOLO_ckpt_path, YOLO_conf_thres
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Define folders
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize YOLO model globally
try:
    model = YOLO(YOLO_ckpt_path)
    app.logger.info("YOLO model loaded successfully")
except Exception as e:
    app.logger.error(f"Error loading YOLO model: {str(e)}")
    model = None


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No image file selected"}), 400

        # Save and process the image
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        # Load image and run detection
        image = Image.open(filepath)
        if model is None:
            return jsonify({"error": "YOLO model not initialized"}), 500

        results = YOLO_detect(model, image, conf_thres=YOLO_conf_thres)

        # Convert results to bounding boxes format
        bboxes = []
        for result in results:
            bbox = {
                "TL_x": float(result[0]),
                "TL_y": float(result[1]),
                "BR_x": float(result[2]),
                "BR_y": float(result[3]),
                "confidence": float(result[4]) if len(result) > 4 else None,
            }
            bboxes.append(bbox)

        # Clean up the uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            app.logger.warning(f"Could not remove temporary file {filepath}: {e}")

        app.logger.info(f"Detection completed. Found {len(bboxes)} objects")
        return jsonify({"filename": filename, "bboxes": bboxes})

    except Exception as e:
        app.logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
