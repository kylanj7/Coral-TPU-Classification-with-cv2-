# EdgeTPU Real-time Image Classifier

A Python application that performs real-time image classification using Google's Edge TPU (Tensor Processing Unit) and a webcam. This script utilizes a quantized MobileNet V2 model optimized for Edge TPU to perform efficient on-device inference.

## Prerequisites

- Coral Edge TPU USB Accelerator
- Compatible operating system (Linux recommended)
- Python 3.7+
- Webcam

## Required Dependencies

```bash
pip install opencv-python
pip install numpy
pip install Pillow
pip install tflite-runtime
```

## Required Files

- `mobilenet_v2_1.0_224_quant_edgetpu.tflite`: Quantized MobileNet V2 model compiled for Edge TPU
- `labels.txt`: Text file containing class labels, one per line
- `libedgetpu.so.1.0`: Edge TPU runtime library

## Features

- Real-time video capture from webcam
- Image preprocessing (resizing and normalization)
- Hardware-accelerated inference using Edge TPU
- Display of top 5 classification results with confidence scores
- Clean shutdown on 'q' key press

## Usage

1. Connect your Edge TPU USB Accelerator
2. Ensure your model and label files are in the same directory as the script
3. Run the script:

```bash
python edge_tpu_classifier.py
```

## Class Description: EdgeTPUClassifier

### Methods

#### `__init__(model_path, label_path)`
- Initializes the Edge TPU delegate and interpreter
- Loads the model and label file
- Sets up input/output tensors

#### `preprocess_image(frame)`
- Converts BGR to RGB color space
- Resizes image to model's required dimensions
- Normalizes pixel values if using a float model

#### `classify_image(frame)`
- Preprocesses the input frame
- Runs inference on Edge TPU
- Returns top 5 predictions with confidence scores

## Main Program Flow

1. Initializes webcam capture
2. Creates EdgeTPUClassifier instance
3. Enters continuous loop:
   - Captures frame from webcam
   - Performs classification
   - Displays results on frame
   - Shows output in window
4. Cleanly releases resources on exit

## Error Handling

- Checks for webcam availability
- Ensures proper resource cleanup in case of errors
- Graceful exit on keyboard interrupt

## Performance Considerations

- Uses hardware acceleration via Edge TPU
- Optimized for real-time processing
- Quantized model for improved efficiency

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Specify your license here]
