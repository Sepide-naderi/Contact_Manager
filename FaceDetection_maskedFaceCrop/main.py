import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Create a Tkinter window
root = tk.Tk()
# Hide the main window
root.withdraw()

file_path = filedialog.askopenfilename(title="Select Image File", filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.jfif"),
                                                                             ("All files", "*.*")))



# Load the pre-trained face detector
detector = dlib.get_frontal_face_detector()
# Load the pre-trained facial landmark detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image = cv2.imread(file_path)

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Assume there is only one face in the image
for face in faces:
    # Detect facial landmarks
    landmarks = predictor(gray, face)

    # Get the convex hull around the face
    points = []
    for i in range(30):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        points.append((x, y))
    hull = cv2.convexHull(np.array(points))

    # Create a mask for the convex hull
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    directory = os.path.dirname('Outputs/')
    filename = os.path.basename(file_path)

    # Generate a new filename
    new_filename = "croppedFace_" + filename

    # Save the masked image
    cv2.imwrite(os.path.join(directory, new_filename), masked_image)

# Display the masked image
cv2.imshow("Masked Face", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


