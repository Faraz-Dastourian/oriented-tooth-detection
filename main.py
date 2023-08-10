import json
import cv2
import numpy as np
import math
import os

# Calculate new coordinates after rotation
def calculate_new_coordinates(old_x, old_y, rotation):
    x = x_top_left
    y = y_top_left

    theta_rad = math.radians(-1 * rotation)

    new_x = x + (old_x - x) * math.cos(theta_rad) + (old_y - y) * math.sin(theta_rad)
    new_y = y - (old_x - x) * math.sin(theta_rad) + (old_y - y) * math.cos(theta_rad)

    return round(new_x), round(new_y)


with open('annotations.json', 'r') as f:
    data = json.load(f)

data_list = os.listdir('./data')




for i in range(len(data_list)):  
    image = cv2.imread(f'./data/{data_list[i]}')

    # Read the original image
    original_image = cv2.imread(f'./data/{data_list[i]}', cv2.IMREAD_GRAYSCALE)

    # Create an empty binary image with the same size as the original image
    binary_image = np.zeros_like(original_image, dtype=np.uint8)

    for label in data[i]['label']:
        img_original_width = label["original_width"]
        img_original_height = label["original_height"]
        rotation = label["rotation"]
        
        width = label["width"] / 100.0 * img_original_width
        height = label["height"] / 100.0 * img_original_height

        x_top_left = label["x"] / 100.0 * img_original_width
        y_top_left = label["y"]/ 100.0 * img_original_height

        x_top_right = x_top_left + width
        y_top_right = y_top_left

        x_bottom_left = x_top_left
        y_bottom_left = y_top_left + height

        x_bottom_right = x_top_left + width
        y_bottom_right = y_top_left + height


        center_x = (x_top_left + x_bottom_right) / 2
        center_y = (y_top_left + y_bottom_right) / 2

        # Calculate the rotated four coorners of the RectangleLabels
        rotated_top_left = calculate_new_coordinates(x_top_left, y_top_left, rotation)
        rotated_top_right = calculate_new_coordinates(x_top_right, y_top_right, rotation)
        rotated_bottom_left = calculate_new_coordinates(x_bottom_left, y_bottom_left, rotation)
        rotated_bottom_right = calculate_new_coordinates(x_bottom_right, y_bottom_right, rotation)

        # Calculated the rotated center coordinates
        rotated_center = np.array(calculate_new_coordinates(center_x,center_y, rotation), np.int32)
        


        points = np.array([rotated_top_left, rotated_top_right, rotated_bottom_right,  rotated_bottom_left],np.int32)
        
        points = points.reshape((-1, 1, 2))



        # Draw the rotated rectangle using cv2.polylines()
        image = cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)


        axesLength = (round(width*0.44), round(height*0.44))
        binary_image = cv2.ellipse(binary_image, rotated_center, axesLength, rotation, 0, 360, color = (255,0,0), thickness = -1)
        image = cv2.ellipse(image, rotated_center, axesLength, rotation, 0, 360, color = (255,0,0), thickness = 2)
        



    # Display the image with the rotated rectangle
    cv2.imwrite(f'./labeled-data/{i+1}-labeled.jpg', image)
    cv2.imwrite(f'./labeled-data/{i+1}-labeled-binary.jpg', binary_image)
    # cv2.imshow('Rotated Rectangle', image)
    # cv2.imshow('Binary Image', binary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
