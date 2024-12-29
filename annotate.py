from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from PIL import Image
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

UPLOAD_FOLDER = "static/uploads"
ANNOTATIONS_FOLDER = "static/annotations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)

BOX_SIZE = 96  # Fixed size for all crops


@app.route("/")
def index():
    return render_template("annotate.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        app.logger.info("Upload request received")

        if "image" not in request.files:
            app.logger.error("No image file in request")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            app.logger.error("No image file selected")
            return jsonify({"error": "No image file selected"}), 400

        # Process CSV if provided
        bboxes = []
        if "csv" in request.files:
            csv_file = request.files["csv"]
            if csv_file.filename != "":
                app.logger.info("Processing CSV file")
                csv_content = csv_file.read().decode("utf-8")
                import csv
                from io import StringIO

                csv_reader = csv.DictReader(StringIO(csv_content))
                for row in csv_reader:
                    try:
                        # Calculate centroid from original coordinates
                        tl_x = float(row["TL_x"])
                        tl_y = float(row["TL_y"])
                        br_x = float(row["BR_x"])
                        br_y = float(row["BR_y"])

                        # Calculate centroid
                        center_x = (tl_x + br_x) / 2
                        center_y = (tl_y + br_y) / 2

                        # Calculate 96x96 box around centroid
                        half_size = BOX_SIZE / 2
                        bbox = {
                            "TL_x": center_x - half_size,
                            "TL_y": center_y - half_size,
                            "BR_x": center_x + half_size,
                            "BR_y": center_y + half_size,
                        }
                        bboxes.append(bbox)
                    except (KeyError, ValueError) as e:
                        app.logger.error(f"Error processing CSV row: {e}")
                        continue

        # Save image file
        from werkzeug.utils import secure_filename

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(filepath)

        app.logger.info(
            f"Files processed successfully. Found {len(bboxes)} bounding boxes"
        )
        return jsonify({"filename": filename, "bboxes": bboxes})

    except Exception as e:
        app.logger.error(f"Error in upload: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/annotate", methods=["POST"])
def annotate():
    try:
        data = request.get_json()
        filename = data.get("filename")
        bboxes = data.get("bboxes", [])

        app.logger.info(f"Annotation request received for {filename}")

        if not filename or not bboxes:
            return jsonify({"error": "Invalid data"}), 400

        image_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404

        img = Image.open(image_path)
        crops = []

        for i, bbox in enumerate(bboxes):
            try:
                # For any input bbox, calculate centroid and create 96x96 box
                if "x" in bbox and "w" in bbox:  # User-drawn format
                    center_x = bbox["x"] + bbox["w"] / 2
                    center_y = bbox["y"] + bbox["h"] / 2
                else:  # Predefined format from CSV
                    center_x = (bbox["TL_x"] + bbox["BR_x"]) / 2
                    center_y = (bbox["TL_y"] + bbox["BR_y"]) / 2

                # Calculate 96x96 box around centroid
                half_size = BOX_SIZE / 2
                x = int(max(0, min(img.width - BOX_SIZE, center_x - half_size)))
                y = int(max(0, min(img.height - BOX_SIZE, center_y - half_size)))

                cropped = img.crop((x, y, x + BOX_SIZE, y + BOX_SIZE))
                crop_filename = f"crop_{i}_{os.path.splitext(filename)[0]}.png"
                crop_path = os.path.join(ANNOTATIONS_FOLDER, crop_filename)
                cropped.save(crop_path)
                crops.append(f"/static/annotations/{crop_filename}")
                app.logger.info(f"Created crop {i}: {crop_path}")

            except Exception as e:
                app.logger.error(f"Error creating crop {i}: {str(e)}")
                continue

        return jsonify({"crops": crops})

    except Exception as e:
        app.logger.error(f"Error in annotate: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8888)
