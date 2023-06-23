from __future__ import print_function
from __future__ import division
import time
import cv2
import numpy as np
import argparse
from math import atan2, cos, sin, sqrt, pi

#Medidas del espacio de trabajo en mm, largo en x y ancho en y
largo = 500
ancho = 500

#Dibujo de los ejes para mostrar la orientacion
def drawAxis(img, p_, q_, colour, scale):
 p = list(p_)
 q = list(q_)
 
 angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
 hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 # Here we lengthen the arrow by a factor of scale
 q[0] = p[0] - scale * hypotenuse * cos(angle)
 q[1] = p[1] - scale * hypotenuse * sin(angle)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 # create the arrow hooks
 p[0] = q[0] + 9 * cos(angle + pi / 4)
 p[1] = q[1] + 9 * sin(angle + pi / 4)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 p[0] = q[0] + 9 * cos(angle - pi / 4)
 p[1] = q[1] + 9 * sin(angle - pi / 4)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)

#Funcion que obtiene la orientacion
def getOrientation(pts, img):
 
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
     data_pts[i,0] = pts[i,0,0]
     data_pts[i,1] = pts[i,0,1]
      # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
 
    return angle

# Function to process each frame
def process_frame(img):

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    k = 7
    # Apply Gaussian blur to the grayscale image
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    lower_thr = 100
    upper_thr = 200
    # Apply Canny edge detection to the blurred image
    edges = cv2.Canny(blur, lower_thr, upper_thr)

    # Find contours in the edges image
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i, c in enumerate(contours):
     # Calculate the area of each contour
     area = cv2.contourArea(c)
     # Ignore contours that are too small or too large
     if area < 0.5e0 or 1e2 < area:
        continue
     # Draw each contour only for visualisation purposes
     cv2.drawContours(img, contours, i, (0, 0, 255), 2)
     # Find the orientation of each shape
     getOrientation(c, img)
     solo_1 = contours[0]
     M = cv2.moments(solo_1)
     if M['m00'] == 0:
        M['m00'] = 1
     Cx = int(M['m10']/M['m00'])
     Cy = int(M['m01'] / M['m00'])
     S = 'Location pixeles' + '(' + str(Cx) + ',' + str(Cy) + ')'
     Px = Cx*largo/640
     Py = Cy*ancho/480
     P = 'Location mm' + '(' + "%.2f" % Px + ',' + "%.2f" % Py + ')'
     font = cv2.FONT_HERSHEY_COMPLEX 
     cv2.putText(img, S, (5, 50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
     cv2.putText(img, P, (5, 450), font, 1, (255, 0, 0), 1, cv2.LINE_AA)



    # Display the processed frame
    cv2.imshow('Processed Frame', img)
    cv2.imshow('Canny Frame', edges)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('toi aca')
        cv2.imwrite(f'Vision/results/detection_{int(time.time())}.png', img)
        return False
    
    return True

# Main function
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        
        # Check if frame reading was successful
        if not ret:
            break
        
        # Process the frame
        if not process_frame(frame):
            break

    
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
if __name__ == '__main__':
    main()
    time.time()