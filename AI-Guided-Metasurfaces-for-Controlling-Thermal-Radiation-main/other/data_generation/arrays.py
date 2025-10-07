from PIL import Image, ImageDraw
import os
import random
import math

# --- Config ---
IMG_SIZE = (256, 256)
BACKGROUND_COLOR_L = 255
SHAPE_COLOR_L = 0

GLOBAL_PADDING = 6       # space between cells
EDGE_MARGIN = 12         # margin from image edge (white space)

BASE_OUTPUT_DIR = "metasurface_dataset_arrays_no_touch"
NUM_ARRAY_IMAGES = 3000

if not os.path.exists(BASE_OUTPUT_DIR):
    os.makedirs(BASE_OUTPUT_DIR)

def save_image(img, subdir, name_prefix, index):
    output_path = os.path.join(BASE_OUTPUT_DIR, subdir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    img_bw = img.convert('1')
    img_bw.save(os.path.join(output_path, f"{name_prefix}_{index:04d}.png"))

def draw_rotated_rectangle(draw_obj, cx, cy, w, h, angle_deg, fill):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    hw, hh = w/2, h/2

    # Calculate rotated corner points
    points = []
    for dx, dy in [(-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)]:
        x = cx + dx*cos_a - dy*sin_a
        y = cy + dx*sin_a + dy*cos_a
        points.append((x,y))
    draw_obj.polygon(points, fill=fill)

def draw_rotated_ellipse(base_img, cx, cy, w, h, angle_deg, fill, bg):
    # create temp canvas big enough to hold rotated ellipse
    max_dim = max(w,h)
    temp_size = int(max_dim * 2)
    temp_img = Image.new('L', (temp_size,temp_size), bg)
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = [temp_size//2 - w//2, temp_size//2 - h//2, temp_size//2 + w//2, temp_size//2 + h//2]
    temp_draw.ellipse(bbox, fill=fill)
    rotated = temp_img.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=bg)
    # paste rotated ellipse onto base image
    px = int(cx - rotated.width//2)
    py = int(cy - rotated.height//2)
    mask = rotated.point(lambda p: 255 if p < 128 else 0)
    base_img.paste(rotated, (px, py), mask)

def get_rotated_bbox_size(w, h, angle_deg):
    """Calculate bounding box of rectangle/ellipse after rotation."""
    angle_rad = math.radians(angle_deg)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    return w*cos_a + h*sin_a, w*sin_a + h*cos_a

def generate_k_arrays_no_touch(num_images):
    subdir = "k_arrays_no_touch_strict"
    if not os.path.exists(os.path.join(BASE_OUTPUT_DIR, subdir)):
        os.makedirs(os.path.join(BASE_OUTPUT_DIR, subdir))

    possible_shapes = ['circle', 'square', 'ellipse', 'bar', 'diamond']

    for img_idx in range(num_images):
        img = Image.new('L', IMG_SIZE, BACKGROUND_COLOR_L)
        draw = ImageDraw.Draw(img)

        # Choose grid size (nx, ny)
        pattern = random.random()
        if pattern < 0.15:
            nx, ny = 3, 3
        elif pattern < 0.5:
            nx, ny = random.choice([(2,2),(3,3)])
        else:
            nx, ny = random.choice([(2,2),(2,3),(3,2)])

        # Compute total grid size inside image with edge margin
        grid_left = EDGE_MARGIN
        grid_top = EDGE_MARGIN
        grid_right = IMG_SIZE[0] - EDGE_MARGIN
        grid_bottom = IMG_SIZE[1] - EDGE_MARGIN

        grid_width = grid_right - grid_left
        grid_height = grid_bottom - grid_top

        # Compute cell sizes including padding
        cell_w = (grid_width - GLOBAL_PADDING*(nx-1)) / nx
        cell_h = (grid_height - GLOBAL_PADDING*(ny-1)) / ny

        # Safety margin inside each cell for shape to avoid touching edges
        cell_inner_margin = min(cell_w, cell_h) * 0.15

        # Max shape size before rotation (width and height)
        max_shape_w = cell_w - 2*cell_inner_margin
        max_shape_h = cell_h - 2*cell_inner_margin

        # To guarantee no overlap when rotated, reduce size by factor
        safe_scale_factor = 0.75

        # Adjust max size accordingly
        max_shape_w *= safe_scale_factor
        max_shape_h *= safe_scale_factor

        for r in range(ny):
            for c in range(nx):
                # Cell top-left corner
                cell_x = grid_left + c * (cell_w + GLOBAL_PADDING)
                cell_y = grid_top + r * (cell_h + GLOBAL_PADDING)

                # Center of cell
                cx = cell_x + cell_w/2
                cy = cell_y + cell_h/2

                # Select shape type randomly
                shape = random.choice(possible_shapes)

                # Random rotation angle for ellipse, bar, diamond; 0 otherwise
                angle = 0
                if shape in ['ellipse', 'bar']:
                    angle = random.uniform(0,179)
                elif shape == 'diamond':
                    shape = 'square'
                    angle = 45

                # Determine shape sizes ensuring bounding box fits cell safely
                # For circle, just radius <= min(max_shape_w, max_shape_h)/2
                if shape == 'circle':
                    radius = min(max_shape_w, max_shape_h)/2
                    radius = int(radius)
                    if radius < 1:
                        radius = 1
                    draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], fill=SHAPE_COLOR_L)

                else:
                    # For rectangle/ellipse/bar, find sizes so rotated bounding box fits max_shape_w x max_shape_h

                    # Start with tentative width and height (max_shape_w/h)
                    # Then reduce so rotated bounding box fits inside max_shape_w/h

                    # Because rotated bbox width = w*cos + h*sin, height = w*sin + h*cos
                    # Solve for w and h such that max rotated bbox <= max_shape_w/h

                    # We'll try to fit width and height equal scaled by aspect ratio
                    aspect_ratio = random.uniform(0.5, 1.5) if shape != 'bar' else random.uniform(2,5)
                    # bars are tall/thin rectangles

                    # Start with max sizes
                    base_w = max_shape_w
                    base_h = max_shape_h

                    # Adjust w and h so rotated bbox fits max_shape_w/h
                    # Iterative approach:
                    w = base_w
                    h = base_h

                    for _ in range(10):
                        rot_w, rot_h = get_rotated_bbox_size(w, h, angle)
                        if rot_w <= base_w and rot_h <= base_h:
                            break
                        # shrink dimensions proportionally
                        w *= 0.9
                        h = w / aspect_ratio

                    # Convert to int and ensure at least 1
                    w = max(1, int(w))
                    h = max(1, int(h))

                    if shape == 'square':
                        w = h = min(w, h)
                        draw_rotated_rectangle(draw, cx, cy, w, h, angle, SHAPE_COLOR_L)
                    elif shape == 'ellipse':
                        draw_rotated_ellipse(img, cx, cy, w, h, angle, SHAPE_COLOR_L, BACKGROUND_COLOR_L)
                    elif shape == 'bar':
                        # Bar: thin rectangle height=h, width=w (w < h)
                        # Already handled aspect_ratio > 1 for bar
                        draw_rotated_rectangle(draw, cx, cy, w, h, angle, SHAPE_COLOR_L)

        save_image(img, subdir, "k_array_strict", img_idx)
        if img_idx % 200 == 0:
            print(f"Generated {img_idx} images...")

    print(f"Done generating '{subdir}' images.")

generate_k_arrays_no_touch(NUM_ARRAY_IMAGES)
