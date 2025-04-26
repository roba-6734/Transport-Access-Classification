import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.Image')



def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess the image for MobileNetV2"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

def predict_image(model_path, img_path, class_indices):
    """Make prediction on a single image"""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess the image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    class_names = {v: k for k, v in class_indices.items()}
    predicted_class_name = class_names[predicted_class]
    
    # Display the image and prediction
    img = image.load_img(img_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class_name}\nConfidence: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
    return predicted_class_name, confidence

if __name__ == "__main__":
    # Path to the trained model
    model_path = 'mobilenet_classifier.keras'
    
    # Class indices from the training output
    class_indices = {'car': 0, 'denied': 1, 'scooter': 2}
    
    # Path to the test image
    test_image_path = 'test/test5.png'  # Fixed path
    
    if not os.path.exists(test_image_path):
        print(f"Error: Image file not found at {test_image_path}")
        print("Please make sure the test image exists in the 'test' directory")
    else:
        try:
            predicted_class, confidence = predict_image(model_path, test_image_path, class_indices)
            print(f"\nPrediction Results:")
            print(f"Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
        except Exception as e:
            print(f"Error during prediction: {str(e)}") 