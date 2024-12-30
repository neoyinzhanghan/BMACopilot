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

        # Convert the results to a serializable format
        detection_results = {
            "filename": filename,
            "raw_output": str(results),  # Convert the raw output to string
            "detections": [],
        }

        # Try to process each detection result
        try:
            for det in results:
                detection = {
                    "class": str(det.cls),
                    "confidence": float(det.conf),
                    "bbox": (
                        det.xywh.tolist() if hasattr(det, "xywh") else det.xyxy.tolist()
                    ),
                }
                detection_results["detections"].append(detection)
        except AttributeError:
            # If the above format doesn't work, try to capture the pandas DataFrame
            try:
                detection_results["detections"] = (
                    results.pandas().xyxy[0].to_dict("records")
                )
            except:
                detection_results["detections"] = str(results)

        # Clean up the uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            app.logger.warning(f"Could not remove temporary file {filepath}: {e}")

        app.logger.info(f"Detection completed")
        return jsonify(detection_results)

    except Exception as e:
        app.logger.error(f"Error in detection: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9999, debug=True)
