from flask import Flask, request, jsonify
import os
import tempfile
import json

app = Flask(__name__)

@app.route('/api/v1/check_drowsiness', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        if 'metadata' not in request.form:
            return jsonify({'error': 'No metadata provided'}), 400

        # Parse JSON metadata correctly
        try:
            json_data = json.loads(request.form["metadata"])
            drowsiness = json_data.get("drowsiness", 70)  # Default to 70 if missing
            face_detect_factor = json_data.get("face_detect_factor", 70)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON format in metadata'}), 400

        # Get the image file
        file = request.files["image"]
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            temp_filename = tmp.name
            file.save(temp_filename)

        # Run detection logic (dummy values for now)
        face_detection_factor = 70 if face_detect_factor is None else face_detect_factor
        drowsiness_factor = 80 if drowsiness is None else drowsiness # Example static value

        response = {
            'received_image': True,
            'drowsiness_factor': drowsiness_factor,
            'face_detection_factor': face_detection_factor,
            'face_detected': face_detection_factor > 70
        }

        # Clean up temporary file
        os.remove(temp_filename)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            'userErrorMessage': 'Failed to parse image',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=False)
