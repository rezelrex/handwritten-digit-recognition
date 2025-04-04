import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and prepare MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create a more robust model
model = tf.keras.models.Sequential([
    # Input layer
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Add Dropout to reduce overfitting
    tf.keras.layers.Dropout(0.2),
    
    # Wider hidden layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    
    # Output layer
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile with same settings
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train for more epochs and SAVE THE HISTORY - THIS IS THE IMPORTANT CHANGE
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# ADD THIS BLOCK RIGHT HERE - after model.fit() and before the image processing
# Get the final validation accuracy (from the last epoch)
final_val_accuracy = history.history['val_accuracy'][-1]
print(f"Final validation accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")

# Evaluate the model explicitly on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Optional: Plot the training and validation accuracy over epochs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Function to preprocess images with multiple approaches
def preprocess_image(img_path, show_steps=True):
    # Read image
    original = cv2.imread(img_path)
    
    # Convert to grayscale if it's color
    if len(original.shape) > 2:
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        gray = original
        
    # Apply threshold to get binary image (removes noise)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to detect the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (should be the digit)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding around the digit
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(binary.shape[1] - x, w + 2*padding)
        h = min(binary.shape[0] - y, h + 2*padding)
        
        # Extract the digit
        digit = binary[y:y+h, x:x+w]
        
        # Resize to 20x20 (MNIST digits are centered in 28x28 with ~4px margins)
        digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Create 28x28 image with the digit centered
        img_centered = np.zeros((28, 28), dtype=np.uint8)
        start_x = (28 - 20) // 2
        start_y = (28 - 20) // 2
        img_centered[start_y:start_y+20, start_x:start_x+20] = digit_resized
        
        # Normalize to [0,1]
        img_normalized = img_centered / 255.0
        
        # Visualize preprocessing steps
        if show_steps:
            fig, axes = plt.subplots(1, 4, figsize=(15, 4))
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original")
            
            axes[1].imshow(binary, cmap='gray')
            axes[1].set_title("Binary (Inverted)")
            
            axes[2].imshow(digit, cmap='gray')
            axes[2].set_title("Extracted Digit")
            
            axes[3].imshow(img_normalized, cmap='gray')
            axes[3].set_title("Centered & Normalized")
            
            plt.tight_layout()
            plt.show()
        
        return img_normalized
    else:
        print(f"No contours found in {img_path}")
        return None

# Process each digit image
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    img_path = f"digits/digit{image_number}.png"
    try:
        # Apply enhanced preprocessing
        img_processed = preprocess_image(img_path)
        
        if img_processed is not None:
            # Reshape for prediction
            img_for_prediction = img_processed.reshape(1, 28, 28)
            
            # Predict
            prediction = model.predict(img_for_prediction, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            print(f"Image {image_number}: This digit is probably a {predicted_digit} (confidence: {confidence:.2f}%)")
            
            # Display comparison with MNIST examples of the same digit
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            
            # Your processed digit
            axes[0].imshow(img_processed, cmap='gray')
            axes[0].set_title(f"Your digit (pred: {predicted_digit})")
            
            # MNIST examples of predicted digit
            mnist_indices = np.where(y_train == predicted_digit)[0]
            if len(mnist_indices) > 0:
                # Show an example matching the prediction
                idx = mnist_indices[0]
                axes[1].imshow(x_train[idx], cmap='gray')
                axes[1].set_title(f"MNIST example of {predicted_digit}")
                
                # Show an example of what it might be instead (different digit)
                for possible_digit in range(10):
                    if possible_digit != predicted_digit:
                        mnist_alt_indices = np.where(y_train == possible_digit)[0]
                        if len(mnist_alt_indices) > 0:
                            idx_alt = mnist_alt_indices[0]
                            axes[2].imshow(x_train[idx_alt], cmap='gray')
                            axes[2].set_title(f"MNIST example of {possible_digit}")
                            break
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error processing image {image_number}: {e}")
    finally:
        image_number += 1