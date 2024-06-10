import cv2
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Define the region of interest (ROI) for hand detection
    roi = frame[100:400, 100:400]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Perform morphological operations to remove noise
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=2)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)

        # Draw the bounding rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Find the convex hull of the largest contour
        hull = cv2.convexHull(max_contour)

        # Draw the convex hull
        cv2.drawContours(roi, [hull], -1, (0, 0, 255), 2)

        # Calculate the convexity defects
        hull_indices = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull_indices)

        # Count the number of defects
        if defects is not None:
            defect_count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Calculate the length of the sides of the triangle
                a = np.linalg.norm(np.array(start) - np.array(end))
                b = np.linalg.norm(np.array(start) - np.array(far))
                c = np.linalg.norm(np.array(end) - np.array(far))

                # Calculate the angle of the defect using the cosine rule
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * (180 / np.pi)

                # Count defects with an angle less than 90 degrees
                if angle <= 90:
                    defect_count += 1
                    cv2.circle(roi, far, 5, (255, 0, 0), -1)

            # Display the detected gesture
            if defect_count == 0:
                gesture = "Fist"
            elif defect_count == 1:
                gesture = "One"
            elif defect_count == 2:
                gesture = "Two"
            elif defect_count == 3:
                gesture = "Three"
            elif defect_count == 4:
                gesture = "Four"
            else:
                gesture = "Open Hand"

            cv2.putText(frame, gesture, (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the ROI and the frame
    cv2.imshow('Thresholded', thresh)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
