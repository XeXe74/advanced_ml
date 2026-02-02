import cv2
import numpy as np
import onnxruntime as ort

# Fashion MNIST Class Labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the ONNX model using ONNX Runtime (More robust than cv2.dnn)
onnx_file = "fashion_mnist_cnn.onnx"
print(f"Loading model from: {onnx_file}")

try:
    # Create an inference session
    session = ort.InferenceSession(onnx_file)
    print("Model loaded successfully with ONNX Runtime!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Get input and output names directly from the model
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define ROI
    h, w, _ = frame.shape
    box_size = 240
    center_x, center_y = w // 2, h // 2
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(w, center_x + box_size // 2)
    y2 = min(h, center_y + box_size // 2)

    roi = frame[y1:y2, x1:x2]

    if roi.size > 0:
        # Preprocessing
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28))

        # Normalize (Pixel - 0.5) / 0.5
        input_data = resized.astype(np.float32) / 255.0
        input_data = (input_data - 0.5) / 0.5

        # Reshape to (1, 1, 28, 28)
        input_tensor = input_data.reshape(1, 1, 28, 28)

        # Run Inference using ONNX Runtime
        # run(output_names, input_feed)
        outputs = session.run([output_name], {input_name: input_tensor})

        # Get Prediction
        # outputs[0] contains the array of probabilities/logits
        prediction_idx = np.argmax(outputs[0])
        label_text = f"{classes[prediction_idx]}"

        # Draw Interface
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pred: {label_text}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Debug View
        debug_view = cv2.resize(resized, (100, 100), interpolation=cv2.INTER_NEAREST)
        frame[0:100, 0:100] = cv2.cvtColor(debug_view, cv2.COLOR_GRAY2BGR)
        cv2.putText(frame, "Input", (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Fashion MNIST Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
