import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define the classes of the traffic signs
classes = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 
    'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
    'No passing', 'No passing for vechiles over 3.5 metric tons', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vechiles', 'Vechiles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
    'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End of no passing by vechiles over 3.5 metric tons'
]

# Define the video capture settings
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BRIGHTNESS, 180)

# Initialize the counter and the current class name
counter = 0
current_class_name = None

# Initialize a flag to keep track of whether the threshold has been reached
threshold_reached = False

current_class = None
class_count = 0

while True:
    # Capture the video frame
    ret, frame = camera.read()

    # Preprocess the image
    img = cv2.resize(frame, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255.0
    img = np.reshape(img, (1, 32, 32, 1))

    # Make the prediction
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    class_name = classes[class_idx]
    prob = round(np.max(predictions) * 100, 2)

    # Check if the predicted class is the same as the current class
    if class_name == current_class:
        class_count += 1
    else:
        current_class = class_name
        class_count = 1

    # Display the result on the image if the current class has been predicted for 5 frames and the probability is greater than 90%
    if class_count >= 10 and prob > 90:
        cv2.putText(frame, f"Class: {class_name}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Probability: {prob}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Traffic Sign Classifier', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
camera.release()
cv2.destroyAllWindows()
