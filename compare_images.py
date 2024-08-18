import cv2
import dlib
import numpy as np

# Load Dlib's pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to detect facial landmarks
def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        return np.array(landmarks_points)
    else:
        return None

# Load the images
img1 = cv2.imread('face1.png')
img2 = cv2.imread('face2.png')

# Detect landmarks
landmarks1 = get_landmarks(img1)
landmarks2 = get_landmarks(img2)

# Check if landmarks are found in both images
if landmarks1 is not None and landmarks2 is not None:
    # Compute the similarity transformation matrix between the landmarks
    H, _ = cv2.estimateAffinePartial2D(landmarks1, landmarks2)

    # Align img1 to img2 based on the transformation matrix
    height, width, channels = img2.shape
    img1_aligned = cv2.warpAffine(img1, H, (width, height))

    # Compute the absolute difference between the aligned face and the reference face
    diff = cv2.absdiff(img2, img1_aligned)

    # Convert to grayscale to simplify thresholding
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to highlight areas of similarity (low difference)
    _, thresholded_diff = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Mask the original image with the thresholded difference to highlight similar areas
    similar_areas = cv2.bitwise_and(img2, img2, mask=thresholded_diff)

    # Save the result
    cv2.imwrite("similar_areas.jpg", similar_areas)

    # Display the results
    cv2.imshow("Similar Areas", similar_areas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Face landmarks could not be detected in one or both images.")
