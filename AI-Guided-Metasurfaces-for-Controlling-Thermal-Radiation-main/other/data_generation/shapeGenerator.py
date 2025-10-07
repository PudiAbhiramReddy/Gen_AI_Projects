from PIL import Image, ImageDraw, ImageFont
import os
import random
import math

# --- Configuration ---
IMG_SIZE = (256, 256)
BACKGROUND_COLOR_L = 255  # White for 'L' mode
SHAPE_COLOR_L = 0         # Black for 'L' mode
NUM_IMAGES_PER_CATEGORY = 500 # Adjust as needed (e.g., 5000 / 12 categories ~= 416)
BASE_OUTPUT_DIR = "metasurface_dataset_specific"
GLOBAL_PADDING = 3 # Min padding from edge for most shapes

# Ensure base directory exists
if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

def save_image(img, subdir_name, name_prefix, index):
    """Saves the image in '1' (binary black and white) mode."""
    output_path = os.path.join(BASE_OUTPUT_DIR, subdir_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Ensure shapes are black (0) and background white (255) before '1' conversion
    # Thresholding helps if any anti-aliasing occurred during rotation/drawing
    final_img_l = img.point(lambda p: SHAPE_COLOR_L if p < 128 else BACKGROUND_COLOR_L)
    img_bw = final_img_l.convert('1')

    img_bw.save(os.path.join(output_path, f"{name_prefix}_{index:04d}.png"))

def get_random_centered_size(min_size_factor=0.1, max_size_factor=0.9):
    """Returns a random size for a shape, intended to be roughly centered."""
    dimension = min(IMG_SIZE) - 2 * GLOBAL_PADDING
    size = random.randint(int(dimension * min_size_factor), int(dimension * max_size_factor))
    return size

def draw_rotated_rectangle(draw_obj, center_x, center_y, width, height, angle_degrees, fill_color):
    """Draws a rotated rectangle using polygon."""
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    points = []
    for x_sign, y_sign in [(-1,-1), (1,-1), (1,1), (-1,1)]:
        # Coords relative to rectangle center
        rx = x_sign * width / 2
        ry = y_sign * height / 2
        # Rotated coords
        rot_x = rx * cos_a - ry * sin_a
        rot_y = rx * sin_a + ry * cos_a
        # Abs coords
        points.append((center_x + rot_x, center_y + rot_y))
    draw_obj.polygon(points, fill=fill_color)

def draw_rotated_ellipse(image_to_draw_on, center_x, center_y, width, height, angle_degrees, fill_color, background_color):
    """Draws a rotated ellipse by creating it on a temp canvas, rotating, then pasting."""
    # Create a temporary image large enough to hold the rotated ellipse
    # The shape is drawn black on white, then rotated
    max_dim = max(width, height)
    temp_size = int(max_dim * 1.5) # Ensure enough space for rotation
    temp_size = max(temp_size,1) # must be >0

    temp_ellipse_img = Image.new('L', (temp_size, temp_size), background_color)
    temp_draw = ImageDraw.Draw(temp_ellipse_img)

    # Draw ellipse centered on temp image
    ellipse_bbox = [
        temp_size // 2 - width // 2,
        temp_size // 2 - height // 2,
        temp_size // 2 + width // 2,
        temp_size // 2 + height // 2
    ]
    temp_draw.ellipse(ellipse_bbox, fill=fill_color)

    rotated_ellipse = temp_ellipse_img.rotate(angle_degrees, resample=Image.BICUBIC, expand=True)

    # Calculate paste position on the main image
    paste_x = center_x - rotated_ellipse.width // 2
    paste_y = center_y - rotated_ellipse.height // 2

    # Paste using the rotated ellipse as a mask to handle transparency/background
    # If fill_color is black (0) and background is white (255),
    # the rotated_ellipse itself can be used as a mask after inverting.
    # For simplicity, we paste directly. Ensure rotated_ellipse background matches.
    # The background_color used in rotate() makes areas outside original bounds match.
    # Pillow paste with 'L' mode mask: 0 is transparent, 255 is opaque.
    # Create a mask from the rotated ellipse (black shape on white bg)
    mask = rotated_ellipse.point(lambda p: 255 if p == fill_color else 0) # Opaque where shape is

    # Create a solid color image of the shape color
    shape_layer = Image.new('L', rotated_ellipse.size, fill_color)

    image_to_draw_on.paste(shape_layer, (paste_x, paste_y), mask=mask)


# --- Category Generators ---

# a) Circles
def generate_a_circles(num_images):
    subdir = "a_circles"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)
        radius = get_random_centered_size(min_size_factor=0.1, max_size_factor=0.9) / 2
        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2 # Centered
        # Slight random offset
        cx += random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2)
        cy += random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2)

        x0 = cx - radius
        y0 = cy - radius
        x1 = cx + radius
        y1 = cy + radius
        draw.ellipse([x0, y0, x1, y1], fill=SHAPE_COLOR_L)
        save_image(img, subdir, "circle", i)
    print(f"Done {subdir}.")

# b) Squares
def generate_b_squares(num_images):
    subdir = "b_squares"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)
        size = get_random_centered_size(min_size_factor=0.1, max_size_factor=0.9)
        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2 # Centered
        # Slight random offset
        cx += random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2)
        cy += random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2)

        x0 = cx - size // 2
        y0 = cy - size // 2
        x1 = cx + size // 2
        y1 = cy + size // 2
        draw.rectangle([x0, y0, x1, y1], fill=SHAPE_COLOR_L)
        save_image(img, subdir, "square", i)
    print(f"Done {subdir}.")

# c) Ellipses (Rotated) - Generator function
def generate_c_ellipses_rotated(num_images):
    subdir = "c_ellipses_rotated"
    print(f"Generating {subdir} (fixed)...")
    if not os.path.exists(os.path.join(BASE_OUTPUT_DIR, subdir)):
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, subdir))

    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L) # Target image with white background

        width = get_random_centered_size(min_size_factor=0.2, max_size_factor=0.9)
        height_ratio = random.uniform(0.3, 0.8)
        height = int(width * height_ratio)
        if random.choice([True, False]):
            width, height = height, width

        width = max(1,width) # ensure positive
        height = max(1,height) # ensure positive

        angle = random.uniform(0, 360)
        # Center with slight jitter
        cx = IMG_SIZE[0] // 2 + random.randint(-GLOBAL_PADDING // 2, GLOBAL_PADDING // 2)
        cy = IMG_SIZE[1] // 2 + random.randint(-GLOBAL_PADDING // 2, GLOBAL_PADDING // 2)

        draw_rotated_ellipse(img, cx, cy, width, height, angle, SHAPE_COLOR_L, BACKGROUND_COLOR_L)
        save_image(img, subdir, "ellipse_rot", i)
    print(f"Done {subdir}.")

# d) Rectangles/Bars (Rotated)
def generate_d_rectangles_rotated(num_images):
    subdir = "d_rectangles_rotated"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        width = get_random_centered_size(min_size_factor=0.2, max_size_factor=0.9)
        height_ratio = random.uniform(0.1, 0.5) # Bar-like
        height = int(width * height_ratio)

        angle = random.uniform(0, 360)
        cx, cy = IMG_SIZE[0] // 2, IMG_SIZE[1] // 2
        cx += random.randint(-GLOBAL_PADDING, GLOBAL_PADDING)
        cy += random.randint(-GLOBAL_PADDING, GLOBAL_PADDING)

        draw_rotated_rectangle(draw, cx, cy, width, height, angle, SHAPE_COLOR_L)
        save_image(img, subdir, "rect_rot", i)
    print(f"Done {subdir}.")

# e) L-Shapes - Generator function
def generate_e_l_shapes(num_images):
    subdir = "e_l_shapes"
    print(f"Generating {subdir} (fixed)...")
    if not os.path.exists(os.path.join(BASE_OUTPUT_DIR, subdir)):
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, subdir))

    for i in range(num_images):
        final_img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)

        thickness = random.randint(max(3, int(IMG_SIZE[0]*0.12)), int(IMG_SIZE[0]*0.25))
        # Arms are length *from the outer corner*
        arm1_outer_len = random.randint(int(IMG_SIZE[0]*0.3), int(IMG_SIZE[0]*0.7))
        arm2_outer_len = random.randint(int(IMG_SIZE[0]*0.3), int(IMG_SIZE[0]*0.7))
        angle = random.choice([0, 90, 180, 270]) + random.uniform(-5, 5) # Slight random rotation

        # Create on a temporary canvas, large enough for rotation
        # Max extent of L-shape before rotation is max(arm1_outer_len, arm2_outer_len)
        temp_canvas_dim = int(max(arm1_outer_len, arm2_outer_len) * 1.5) # Diagonal for safety
        temp_canvas_dim = max(temp_canvas_dim, 1)

        temp_img = Image.new('L', (temp_canvas_dim, temp_canvas_dim), BACKGROUND_COLOR_L)
        temp_draw = ImageDraw.Draw(temp_img)

        # Draw L-shape centered within this temporary canvas
        # This helps ensure the "center of mass" of the L is near the rotation center.
        tc_cx, tc_cy = temp_canvas_dim // 2, temp_canvas_dim // 2

        # Define points for the L-shape polygon for more robust rotation
        # Example: L with corner at (tc_cx, tc_cy), extending up and right
        # This is tricky. Simpler: draw two overlapping rectangles.
        # Let's draw it with origin at the *inner corner* for simplicity, then offset.

        # Rect 1 (vertical arm)
        # Assume arm1 is vertical, arm2 is horizontal, corner at bottom-left of this setup
        # Draw it relative to a local origin, then shift to center of temp_img
        local_origin_x = tc_cx - arm2_outer_len / 2  # Crude centering
        local_origin_y = tc_cy + arm1_outer_len / 2

        # Vertical arm
        # (x0, y0, x1, y1)
        r1_x0 = local_origin_x
        r1_y0 = local_origin_y - arm1_outer_len
        r1_x1 = local_origin_x + thickness
        r1_y1 = local_origin_y
        temp_draw.rectangle([r1_x0, r1_y0, r1_x1, r1_y1], fill=SHAPE_COLOR_L)

        # Horizontal arm
        r2_x0 = local_origin_x
        r2_y0 = local_origin_y - thickness # Top of horizontal arm aligns with bottom of L
        r2_x1 = local_origin_x + arm2_outer_len
        r2_y1 = local_origin_y
        temp_draw.rectangle([r2_x0, r2_y0, r2_x1, r2_y1], fill=SHAPE_COLOR_L)

        # Rotate the temp image
        rotated_l_img = temp_img.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=BACKGROUND_COLOR_L)

        # Paste onto main image, centering the bounding box of rotated_l_img
        paste_cx = IMG_SIZE[0] // 2 + random.randint(-GLOBAL_PADDING // 2, GLOBAL_PADDING // 2)
        paste_cy = IMG_SIZE[1] // 2 + random.randint(-GLOBAL_PADDING // 2, GLOBAL_PADDING // 2)

        paste_x = paste_cx - rotated_l_img.width // 2
        paste_y = paste_cy - rotated_l_img.height // 2

        mask = rotated_l_img.point(lambda p: 255 if p == SHAPE_COLOR_L else 0)
        final_img.paste(rotated_l_img, (paste_x, paste_y), mask=mask)

        save_image(final_img, subdir, "l_shape", i)
    print(f"Done {subdir}.")

# f) C-Shapes/Split Rings (Curved)
def generate_f_c_shapes_curved(num_images):
    subdir = "f_c_shapes_curved"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        cx, cy = IMG_SIZE[0] // 2 + random.randint(-5,5), IMG_SIZE[1] // 2 + random.randint(-5,5)
        outer_radius = get_random_centered_size(min_size_factor=0.3, max_size_factor=0.8) / 2
        thickness_ratio = random.uniform(0.2, 0.5)
        thickness = max(2, int(outer_radius * thickness_ratio))
        inner_radius = outer_radius - thickness

        if inner_radius < 1: inner_radius = 1 # ensure valid

        gap_angle_degrees = random.uniform(30, 120)
        start_angle_offset = random.uniform(0, 360) # Orientation of the gap

        start_angle = gap_angle_degrees / 2 + start_angle_offset
        end_angle = 360 - gap_angle_degrees / 2 + start_angle_offset

        # Draw outer arc
        draw.arc([cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius],
                 start_angle, end_angle, fill=SHAPE_COLOR_L, width=thickness)

        save_image(img, subdir, "c_curved", i)
    print(f"Done {subdir}.")

# g) Arcs/Wedges/Pac-Man
def generate_g_arcs_wedges(num_images):
    subdir = "g_arcs_wedges"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        cx, cy = IMG_SIZE[0] // 2 + random.randint(-5,5), IMG_SIZE[1] // 2 + random.randint(-5,5)
        radius = get_random_centered_size(min_size_factor=0.25, max_size_factor=0.9) / 2

        start_angle = random.uniform(0, 360)
        # Make sure end_angle is different from start_angle for a visible wedge
        # Arc length from 30 to 330 degrees (to avoid full circle or tiny sliver)
        arc_length = random.uniform(30, 330)
        end_angle = start_angle + arc_length

        draw.pieslice([cx - radius, cy - radius, cx + radius, cy + radius],
                      start_angle, end_angle, fill=SHAPE_COLOR_L)
        save_image(img, subdir, "wedge", i)
    print(f"Done {subdir}.")

# h) Crosses
def generate_h_crosses(num_images):
    subdir = "h_crosses"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        cx, cy = IMG_SIZE[0] // 2 + random.randint(-5,5), IMG_SIZE[1] // 2 + random.randint(-5,5)

        # Create two bars (rectangles)
        len1 = get_random_centered_size(0.3, 0.9)
        thick1 = random.randint(max(2, int(len1*0.1)), int(len1*0.3))
        angle1 = random.uniform(0, 180) # First bar angle

        len2 = get_random_centered_size(0.3, 0.9)
        thick2 = random.randint(max(2, int(len2*0.1)), int(len2*0.3))
        # Second bar angle is roughly perpendicular or offset
        angle2 = angle1 + 90 + random.uniform(-30, 30)

        draw_rotated_rectangle(draw, cx, cy, len1, thick1, angle1, SHAPE_COLOR_L)
        draw_rotated_rectangle(draw, cx, cy, len2, thick2, angle2, SHAPE_COLOR_L)

        save_image(img, subdir, "cross", i)
    print(f"Done {subdir}.")


# j) Polygons (Triangle, Hexagon, etc.)
def generate_j_polygons(num_images):
    subdir = "j_polygons"
    print(f"Generating {subdir}...")
    for i in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        cx, cy = IMG_SIZE[0] // 2 + random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2), \
                 IMG_SIZE[1] // 2 + random.randint(-GLOBAL_PADDING*2, GLOBAL_PADDING*2)

        radius = get_random_centered_size(min_size_factor=0.2, max_size_factor=0.85) / 2
        num_vertices = random.choice([3, 4, 5, 6, 7, 8]) # Tri, Sqr, Pent, Hex, Hept, Oct

        angle_offset = random.uniform(0, 360) # Random orientation

        points = []
        for k in range(num_vertices):
            angle = math.radians(360 * k / num_vertices + angle_offset)
            # Add some randomness to radius for irregular polygons too
            current_radius = radius * random.uniform(0.85, 1.15) if num_vertices > 4 else radius
            x = cx + current_radius * math.cos(angle)
            y = cy + current_radius * math.sin(angle)
            points.append((x, y))

        draw.polygon(points, fill=SHAPE_COLOR_L)
        save_image(img, subdir, "polygon", i)
    print(f"Done {subdir}.")

# --- Main Execution ---
if __name__ == "__main__":
    generators_map = {
        'a': generate_a_circles,
        'b': generate_b_squares,
        'd': generate_d_rectangles_rotated,
        'e': generate_e_l_shapes,
        'f': generate_f_c_shapes_curved,
        'g': generate_g_arcs_wedges,
        'h': generate_h_crosses,
        'j': generate_j_polygons,

    }

    total_generated = 0
    for cat_label, gen_func in generators_map.items():
        print(f"\nProcessing Category: {cat_label}")
        gen_func(NUM_IMAGES_PER_CATEGORY)
        total_generated += NUM_IMAGES_PER_CATEGORY

    print(f"\n--- All Categories Done! ---")
    print(f"Total images generated: {total_generated}")
    print(f"Dataset saved in directory: {BASE_OUTPUT_DIR}")

import shutil
