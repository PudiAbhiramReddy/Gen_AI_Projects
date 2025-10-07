import os
import cv2
import numpy as np
import random
import math

# Directory to save the images
output_dir = 'arrays'
os.makedirs(output_dir, exist_ok=True)

# Image and shape settings
IMG_SIZE = 256
PADDING = 10
SHAPE_MIN_SIZE = 20
SHAPE_MAX_SIZE = 40

# List of base geometric shapes
def draw_shape(shape_type, size):
    canvas = np.zeros((size, size), dtype=np.uint8)
    center = size // 2
    if shape_type == "circle":
        cv2.circle(canvas, (center, center), size // 2 - 2, 255, -1)
    elif shape_type == "square":
        cv2.rectangle(canvas, (2, 2), (size - 2, size - 2), 255, -1)
    elif shape_type == "triangle":
        pts = np.array([[center, 2], [2, size - 2], [size - 2, size - 2]], np.int32)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "pentagon":
        pts = regular_polygon(center, center, size // 2 - 2, 5)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "hexagon":
        pts = regular_polygon(center, center, size // 2 - 2, 6)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "heptagon":
        pts = regular_polygon(center, center, size // 2 - 2, 7)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "octagon":
        pts = regular_polygon(center, center, size // 2 - 2, 8)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "rhombus":
        pts = np.array([[center, 2], [size - 2, center], [center, size - 2], [2, center]], np.int32)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    elif shape_type == "parallelogram":
        pts = np.array([[6, 2], [size - 2, 2], [size - 6, size - 2], [2, size - 2]], np.int32)
        cv2.drawContours(canvas, [pts], 0, 255, -1)
    return canvas

def regular_polygon(xc, yc, radius, sides):
    pts = []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        x = int(xc + radius * math.cos(angle))
        y = int(yc + radius * math.sin(angle))
        pts.append([x, y])
    return np.array(pts, np.int32)

SHAPE_TYPES = [
    "circle", "square", "triangle", "pentagon", "hexagon", 
    "heptagon", "octagon", "rhombus", "parallelogram"
]

def generate_array_image():
    img = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    # Random rows and cols (e.g. 2–5 each)
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)

    spacing_x = (IMG_SIZE - 2 * PADDING) // cols
    spacing_y = (IMG_SIZE - 2 * PADDING) // rows

    for r in range(rows):
        for c in range(cols):
            size = random.randint(SHAPE_MIN_SIZE, SHAPE_MAX_SIZE)
            shape = random.choice(SHAPE_TYPES)
            shape_img = draw_shape(shape, size)

            # Random rotation
            angle = random.randint(0, 360)
            rot_mat = cv2.getRotationMatrix2D((size//2, size//2), angle, 1.0)
            rotated = cv2.warpAffine(shape_img, rot_mat, (size, size), borderValue=0)

            # Positioning
            x = PADDING + c * spacing_x + spacing_x // 2 - size // 2
            y = PADDING + r * spacing_y + spacing_y // 2 - size // 2

            if 0 <= x <= IMG_SIZE - size and 0 <= y <= IMG_SIZE - size:
                roi = img[y:y+size, x:x+size]
                mask = rotated > 0
                roi[mask] = 0

    return img

# Number of images to generate
num_images = 10  # Change this to however many you want

for i in range(num_images):
    array_img = generate_array_image()
    cv2.imwrite(os.path.join(output_dir, f'array_{i:03d}.png'), array_img)
