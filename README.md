# Handwritten Digit Recognition Project

## 1. Introduction

**Problem Statement and Relevance:**
This project tackles the classic problem of handwritten digit recognition, a fundamental task in computer vision and a core application of machine learning. Accurate digit recognition has significant real-world utility in areas such as postal automation, bank check processing, and data entry from forms.

**Motivation:**
The primary motivation behind this project was to gain hands-on experience in applying neural network concepts, specifically focusing on a feedforward architecture, to an image classification problem. A secondary goal was to evaluate the trained model's ability to generalize beyond the standard MNIST dataset by testing it on personally created handwritten digit images, adding a layer of personalized evaluation.

**Related Course Topics:**
This project directly aligns with the course topics of **Artificial Neural Networks and Deep Learning**. The implementation employs a feedforward neural network architecture, incorporating dense layers and dropout, which are key concepts in deep learning. The project also utilizes the widely recognized MNIST dataset, further connecting it to established practices and benchmarks in the field.

## 2. Methodology

**Chosen Machine Intelligence Approach:**
A feedforward neural network model was selected for this project and implemented using the TensorFlow/Keras library in Python. This approach was chosen for its capacity to learn complex patterns from data and its common application in image classification tasks, serving as a solid foundation before exploring more advanced architectures like Convolutional Neural Networks (CNNs).

**Justification of Algorithms and Models:**
The implemented model consists of the following layers:
* An input `Flatten` layer to convert the 28x28 pixel images into a 784-dimensional vector.
* Two dense hidden layers with ReLU (Rectified Linear Unit) activation functions, containing 256 and 128 units respectively. ReLU introduces non-linearity, enabling the network to learn intricate relationships within the data.
* A `Dropout` layer with a rate of 0.2, applied after the first hidden layer. Dropout is a regularization technique used to prevent overfitting by randomly setting a fraction of the input units to 0 during training.
* A final dense output layer with 10 units (corresponding to the 10 digits from 0 to 9) and a `softmax` activation function. Softmax converts the output of the last layer into a probability distribution over the 10 classes, ensuring that the predicted probabilities for each digit sum up to one.

**Dataset Used and Preprocessing Steps:**
The project utilizes the **MNIST dataset**, which comprises 70,000 grayscale images of handwritten digits. This dataset is split into 60,000 images for training and 10,000 for testing. The dataset was conveniently loaded using the `tensorflow.keras.datasets.mnist` module. The pixel values of the images, originally in the range [0, 255], were normalized by dividing each value by 255.0. This scaling to the range [0, 1] is crucial for improving the training speed and stability of the neural network.

In addition to using the MNIST dataset, a custom image preprocessing function, `preprocess_image`, was developed using the OpenCV (cv2) library to handle user-drawn digit images. This function performs the following steps:
* Converts the input image to grayscale.
* Applies binary thresholding (using `cv2.THRESH_BINARY_INV`) to create a black and white image where the digit is white on a black background.
* Finds all contours in the binary image using `cv2.findContours`.
* Identifies the largest contour, assuming it represents the handwritten digit.
* Extracts the bounding box of the largest contour.
* Adds a padding of 5 pixels around the bounding box.
* Extracts the region of interest (the digit with padding) from the binary image.
* Resizes the extracted digit to a 20x20 pixel size using `cv2.INTER_AREA` interpolation.
* Creates a new 28x28 black image and centers the 20x20 resized digit within it, mimicking the structure of the MNIST images.
* Normalizes the pixel values of the centered 28x28 image to the range [0, 1] by dividing by 255.0.

## 3. Implementation

**Concise Technical Explanation:**
The handwritten digit recognition model was built using the Keras API within the TensorFlow framework. The `Sequential` API allowed for the creation of the model by stacking layers in a linear fashion. The training process involved compiling the model with the `adam` optimizer, the `sparse_categorical_crossentropy` loss function (suitable for integer-labeled classification), and `accuracy` as the metric to monitor performance. The model was trained for 20 epochs using the MNIST training data. The `validation_data` parameter was used during training to evaluate the model's performance on the MNIST test set after each epoch, providing insights into its generalization capabilities.

A separate Python function, `preprocess_image`, was implemented using the OpenCV library to handle custom digit images. This function utilizes image processing techniques such as grayscale conversion, binary thresholding, and contour detection to isolate and prepare the handwritten digit for prediction. Once processed, the image is reshaped to match the expected input shape of the trained model (1x28x28) and passed to the `model.predict()` method. The predicted digit is then determined by finding the index of the highest probability in the output of the softmax layer using `np.argmax()`.

**Software Developed:**
The primary software developed for this project is the Python script containing the code for model training, custom image preprocessing, and prediction.

**Software Tools, Frameworks, and Programming Languages Used:**
* **Python:** The primary programming language used (version 3.12.7).
* **TensorFlow (with Keras):** A powerful open-source library for numerical computation and large-scale machine learning. Keras is TensorFlow's high-level API for building and training neural networks.
* **NumPy:** A fundamental package for numerical computation in Python, used for array manipulation and mathematical operations.
* **OpenCV (cv2):** A comprehensive library for computer vision tasks, used here for image preprocessing of custom drawn digits.
* **Matplotlib:** A plotting library in Python used for visualizing the image preprocessing steps and displaying images.
* **os:** A module providing a way of using operating system-dependent functionality, used for handling file paths.

**Challenges Encountered and Solutions Applied:**
A significant challenge was ensuring the consistency of the format between the MNIST dataset and the custom drawn digit images. Initially, the model exhibited poor performance on the custom images. To address this, a more robust preprocessing pipeline was implemented using OpenCV. This involved carefully applying thresholding to obtain a clean binary image, accurately identifying the digit contour, and then resizing and centering the digit within a 28x28 canvas to align with the MNIST image structure. Visualizing the intermediate steps of the preprocessing using Matplotlib proved invaluable for debugging and refining the pipeline. Another potential challenge was overfitting, which was mitigated by the inclusion of a `Dropout` layer in the model architecture and by monitoring the validation accuracy during the training process.

## 4. Results and Discussion

**Present Experimental Results:**
The feedforward neural network was trained for 20 epochs. The validation accuracy on the MNIST test set reached a final value of **0.9813 (98.13%)**. Following training, the `preprocess_image` function was used to process a series of nine custom drawn digit images (named `digit1.png` through `digit9.png`). The model's predictions on these images showed varying degrees of accuracy:

* **Correctly Classified:** Digit 1 (likely '1'), Digit 2 (likely '2'), Digit 5 (likely '5'), Digit 8 (likely '8'). These were generally predicted with higher confidence.
* **Misclassified:** Digits 3, 4, 6, 7, and 9 were often incorrectly classified, frequently as '8' or '3', with lower confidence scores compared to the correctly classified digits.

For each processed custom image, visualizations were generated using Matplotlib to show the original image, the binary (inverted) image, the extracted digit, and the final centered and normalized image. Additionally, comparisons were made by displaying an example from the MNIST training set that matched the model's prediction and an example of a likely alternative digit.

**Analyze the Effectiveness of the Approach:**
The high validation accuracy achieved on the MNIST test set indicates that the feedforward neural network is effective at learning the patterns within the standardized MNIST dataset. However, the inconsistent performance on the custom drawn digits suggests that the model's generalization ability to more varied and less standardized handwriting styles is limited. Factors contributing to this discrepancy could include:

* Differences in drawing style and thickness of the digit strokes.
* Variations in image quality and background noise in the custom images.
* The relative simplicity of the feedforward neural network architecture compared to more advanced models like CNNs, which are specifically designed to handle spatial hierarchies in image data.

**Compare Results with Alternative Techniques or Benchmarks:**
While a direct comparison with other techniques was not the primary focus of this project, it is well-established that Convolutional Neural Networks (CNNs) typically achieve state-of-the-art results on image classification tasks, including MNIST, often surpassing the accuracy of simple feedforward networks. Future work could involve implementing and evaluating a CNN model on both the MNIST dataset and the custom drawn digit images to assess the potential for improved performance.

## 5. Conclusion and Future Work

**Summarize Key Findings and Insights:**
This project successfully implemented a feedforward neural network for the task of handwritten digit recognition. The model demonstrated strong performance on the MNIST dataset, achieving a high validation accuracy. However, the evaluation on custom drawn digits highlighted the challenges of generalizing to real-world variations in handwriting and image quality. The importance of careful image preprocessing in bridging the gap between standardized training data and real-world inputs was also underscored.

**Discuss Ethical Considerations, Risks, and/or Limitations:**
While this project is primarily educational, the broader applications of OCR technology have ethical implications, particularly concerning privacy when processing personal documents. Potential biases in recognition accuracy across different handwriting styles could also lead to unfair outcomes in certain applications. The limitations of this project include the use of a relatively simple feedforward network architecture, which might not be as robust as more complex models like CNNs. The model's performance is also likely sensitive to the quality and style of the input handwritten digits.

**Suggest Improvements or Extensions for Future Research:**
* **Implement a Convolutional Neural Network (CNN) model:** CNNs are better suited for image recognition tasks and are expected to yield higher accuracy.
* **Train on a more diverse dataset:** Expanding the training data to include more variations in handwriting styles, potentially by collecting custom data, could improve generalization.
* **Explore data augmentation techniques:** Applying transformations like slight rotations, shifts, and scaling to the training images can help the model become more robust to variations in the input.
* **Implement more sophisticated image preprocessing techniques:** Further refinement of the preprocessing steps for custom images could improve the accuracy of recognition.
* **Investigate transfer learning:** Explore the use of pre-trained models on larger image datasets and fine-tune them for handwritten digit recognition.
* **Develop a user interface:** Creating a simple graphical user interface (GUI) would allow for a more interactive experience where users can draw digits and see the model's predictions in real-time.

## 6. References

* TensorFlow documentation: [https://www.tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
* Keras documentation: [https://www.tensorflow.org/guide/keras](https://www.tensorflow.org/guide/keras)
* MNIST dataset: [https://www.kaggle.com/datasets/hojjatk/mnist-dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
* OpenCV documentation: [https://docs.opencv.org/4.x/index.html](https://docs.opencv.org/4.x/index.html)
* Youtube Tutorial: [https://www.youtube.com/watch?v=bte8Er0QhDg](https://www.youtube.com/watch?v=bte8Er0QhDg) *(Note: You might want to replace this with the actual tutorial link if you used one)*
