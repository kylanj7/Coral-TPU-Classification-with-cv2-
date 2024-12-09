import cv2
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate
import platform

class EdgeTPUClassifier:
    def __init__(self, model_path, label_path):
        # Initialize Edge TPU delegate
        delegate = load_delegate('libedgetpu.so.1.0')
        self.interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=[delegate]
        )
        self.interpreter.allocate_tensors()
        
        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        
        # Load labels
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize and normalize
        image = Image.fromarray(rgb_frame)
        image = image.resize((self.width, self.height))
        input_data = np.expand_dims(image, axis=0)
        
        # Normalize pixel values if using a float model (aka "1.0")
        if self.input_details[0]['dtype'] == np.float32:
            input_data = (np.float32(input_data) - 127.5) / 127.5
            
        return input_data
    
    def classify_image(self, frame):
        input_data = self.preprocess_image(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get prediction results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        top_k = output_data[0].argsort()[-5:][::-1]  # Get top 5 predictions
        
        results = []
        for idx in top_k:
            score = float(output_data[0][idx])
            results.append((self.labels[idx], score))
            
        return results

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
        
    # Initialize classifier
    model_path = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    label_path = 'labels.txt'
    classifier = EdgeTPUClassifier(model_path, label_path)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get predictions
            results = classifier.classify_image(frame)
            
            # Display results
            y_pos = 30
            for label, score in results:
                text = f'{label}: {score:.2f}'
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_pos += 30
                
            cv2.imshow('Real-time Classification', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
