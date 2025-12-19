import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
import io

# ==========================================
# 1. CORE IMAGE PROCESSING FUNCTIONS
# ==========================================

def tile_pattern(pattern_img, target_shape, pattern_ppi, target_ppi):
    """
    Standard tiling for 0-degree or 90-degree rotations.
    """
    tgt_h, tgt_w = target_shape[:2]
    if pattern_ppi <= 0: pattern_ppi = 1
    
    scale = target_ppi / pattern_ppi
    new_w = int(pattern_img.shape[1] * scale)
    new_h = int(pattern_img.shape[0] * scale)
    
    if new_w < 1 or new_h < 1: return np.zeros(target_shape, dtype=np.uint8)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_pattern = cv2.resize(pattern_img, (new_w, new_h), interpolation=interp)
    
    tiles_x = (tgt_w // new_w) + 2 
    tiles_y = (tgt_h // new_h) + 2
    
    tiled = np.tile(resized_pattern, (tiles_y, tiles_x, 1))
    return tiled[:tgt_h, :tgt_w]

def generate_angled_texture(pattern_img, target_shape, pattern_ppi, target_ppi, angle):
    """
    Advanced Tiling: Handles arbitrary angles (e.g., 25 degrees).
    Strategy: Tile a huge canvas -> Rotate the Canvas -> Crop center.
    """
    tgt_h, tgt_w = target_shape[:2]
    
    # 1. If angle is simple (0, 90, 180, -90), use the fast simple tiler
    if angle == 0:
        return tile_pattern(pattern_img, target_shape, pattern_ppi, target_ppi)
    elif angle == 90:
        rot = cv2.rotate(pattern_img, cv2.ROTATE_90_CLOCKWISE)
        return tile_pattern(rot, target_shape, pattern_ppi, target_ppi)
    elif angle == -90 or angle == 270:
        rot = cv2.rotate(pattern_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return tile_pattern(rot, target_shape, pattern_ppi, target_ppi)
    elif angle == 180:
        rot = cv2.rotate(pattern_img, cv2.ROTATE_180)
        return tile_pattern(rot, target_shape, pattern_ppi, target_ppi)

    # 2. Complex Angle Logic (e.g. 25 degrees)
    # We need a canvas big enough to hold the rotated image without showing edges
    # The diagonal length is the safest size
    diagonal = int(np.sqrt(tgt_h**2 + tgt_w**2))
    padding = int(diagonal * 0.2) # Add 20% padding
    canvas_size = (diagonal + padding, diagonal + padding)
    
    # Tile onto the HUGE canvas (at 0 degrees)
    big_tiled = tile_pattern(pattern_img, canvas_size, pattern_ppi, target_ppi)
    
    # 3. Rotate the Huge Canvas
    center = (canvas_size[1] // 2, canvas_size[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Warp with Border Wrap to ensure seamless edges if we somehow hit the edge
    rotated_big = cv2.warpAffine(big_tiled, M, (canvas_size[1], canvas_size[0]), 
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # 4. Crop the Center (matching the target view size)
    start_x = center[0] - tgt_w // 2
    start_y = center[1] - tgt_h // 2
    
    # Safety Check bounds
    if start_x < 0: start_x = 0
    if start_y < 0: start_y = 0
    
    cropped = rotated_big[start_y : start_y+tgt_h, start_x : start_x+tgt_w]
    
    # Double check size (sometimes rounding errors occur)
    if cropped.shape[:2] != (tgt_h, tgt_w):
        cropped = cv2.resize(cropped, (tgt_w, tgt_h))
        
    return cropped

def displacement_warp(pattern_layer, depth_map, base_img_gray, strength=30):
    if depth_map is None: return pattern_layer
    rows, cols = pattern_layer.shape[:2]

    depth = cv2.resize(depth_map, (cols, rows)).astype(np.float32)

    if base_img_gray is None:
        base_detail = np.zeros((rows, cols), dtype=np.float32)
    else:
        base_detail = cv2.resize(base_img_gray, (cols, rows)).astype(np.float32)
        base_detail = 255.0 - base_detail

    gx_d = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy_d = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    
    gx_b = cv2.Sobel(base_detail, cv2.CV_32F, 1, 0, ksize=3)
    gy_b = cv2.Sobel(base_detail, cv2.CV_32F, 0, 1, ksize=3)

    gx = (gx_d * 0.4) + (gx_b * 0.6)
    gy = (gy_d * 0.4) + (gy_b * 0.6)

    x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    
    map_x = (x_grid + (gx / 255.0) * strength).astype(np.float32)
    map_y = (y_grid + (gy / 255.0) * strength).astype(np.float32)

    return cv2.remap(pattern_layer, map_x, map_y, 
                     interpolation=cv2.INTER_LINEAR, 
                     borderMode=cv2.BORDER_REFLECT)

def blend_texture(base_img_rgb, mask, texture_rgb, brightness=1.0):
    if mask is None: return base_img_rgb
    
    if texture_rgb.shape[:2] != base_img_rgb.shape[:2]:
        texture_rgb = cv2.resize(texture_rgb, (base_img_rgb.shape[1], base_img_rgb.shape[0]))
    
    # Shadow Mapping
    gray_base = cv2.cvtColor(base_img_rgb, cv2.COLOR_RGB2GRAY).astype(float) / 255.0
    
    if mask.shape[:2] != gray_base.shape[:2]:
        mask = cv2.resize(mask, (gray_base.shape[1], gray_base.shape[0]))
        
    masked_pixels = gray_base[mask > 128] 
    if len(masked_pixels) > 0:
        white_point = np.percentile(masked_pixels, 80)
        if white_point < 0.1: white_point = 0.5 
    else:
        white_point = 1.0

    shading = gray_base / white_point
    shading = np.clip(shading, 0, 1.0)
    shading = np.power(shading, 0.8) # Gamma
    shading = shading * shading * (3 - 2 * shading) # Contrast
    shading = shading * brightness
    shading = np.clip(shading, 0, 1.0)
    shading_stack = np.stack([shading]*3, axis=-1)

    # Specular
    highlights = gray_base - 0.7 
    highlights = np.clip(highlights, 0, 1.0)
    highlights = np.power(highlights, 2.0) 
    highlights = highlights * 0.4 
    highlight_stack = np.stack([highlights]*3, axis=-1)

    # Blend
    tex_float = texture_rgb.astype(float) / 255.0
    composite = tex_float * shading_stack
    composite = composite + highlight_stack
    
    # Grain
    noise = np.random.normal(0, 0.02, composite.shape).astype(np.float32)
    composite = composite + noise
    composite = np.clip(composite, 0, 1.0)
    
    # Mask
    mask_float = mask.astype(float) / 255.0
    if len(mask_float.shape) == 2:
        mask_float = np.stack([mask_float]*3, axis=-1)

    base_float = base_img_rgb.astype(float) / 255.0
    final = (composite * mask_float + base_float * (1 - mask_float))
    
    return np.clip(final * 255, 0, 255).astype(np.uint8)

def load_asset(folder, name_no_ext, grayscale=False):
    for ext in [".png", ".jpg", ".jpeg"]:
        full_path = os.path.join(folder, name_no_ext + ext)
        if os.path.exists(full_path):
            flags = 0 if grayscale else 1
            return cv2.imread(full_path, flags)
    return None

# ==========================================
# 2. STREAMLIT FRONTEND
# ==========================================

st.set_page_config(page_title="Bedding V-TON", layout="wide")
st.title("Bedding Virtual Try-On POC")

with st.sidebar:
    st.header("Step 1: Upload Pattern")
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    st.header("Step 2: Dimensions")
    repeat_size_inches = st.number_input("Pattern Real Size (Inches)", min_value=1.0, value=10.0, step=0.5)
    
    st.info("Ensure your 'templates' folder contains: headon, side, quarter, overhead")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    pattern_bgr = cv2.imdecode(file_bytes, 1)
    pattern_rgb = cv2.cvtColor(pattern_bgr, cv2.COLOR_BGR2RGB)
    
    st.sidebar.image(pattern_rgb, caption="Uploaded Pattern", use_container_width=True)
    
    if st.sidebar.button("Run Simulation"):
        st.write("Processing Views...")
        
        pat_h, pat_w = pattern_rgb.shape[:2]
        pattern_ppi = pat_w / repeat_size_inches
        
        view_folders = ["headon", "quarter", "side", "overhead"]
        cols = st.columns(2)
        
        for idx, view_name in enumerate(view_folders):
            base_path = os.path.join("templates", view_name)
            if not os.path.exists(base_path): continue 
                
            try:
                base_img_bgr = load_asset(base_path, "base", grayscale=False)
                if base_img_bgr is None: continue
                base_img = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2RGB)
                base_img_gray = cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2GRAY)

                mask_comforter = load_asset(base_path, "mask_comforter", grayscale=True)
                mask_shams = load_asset(base_path, "mask_shams", grayscale=True)
                depth_map = load_asset(base_path, "depth", grayscale=True)
                
                config_path = os.path.join(base_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f: config = json.load(f)
                else:
                    config = {}

                # Config Values
                conf_brightness = config.get('brightness', 1.0)
                disp_str = config.get('displacement_strength', 30)
                
                # --- NEW ROTATION LOGIC ---
                angle_conf = config.get('rotation', 0)
                angle_sham = config.get('sham_rotation', 0)

                # Process Comforter
                if mask_comforter is not None:
                    # Use the new "Angled Texture" generator
                    tiled_conf = generate_angled_texture(pattern_rgb, base_img.shape, pattern_ppi, config.get('comforter_ppi', 20), angle_conf)
                    
                    warped_conf = displacement_warp(tiled_conf, depth_map, base_img_gray, strength=disp_str)
                    result = blend_texture(base_img, mask_comforter, warped_conf, brightness=conf_brightness)
                else:
                    result = base_img.copy()

                # Process Shams
                if mask_shams is not None:
                    tiled_sham = generate_angled_texture(pattern_rgb, base_img.shape, pattern_ppi, config.get('sham_ppi', 20), angle_sham)
                    
                    warped_sham = displacement_warp(tiled_sham, depth_map, base_img_gray, strength=disp_str)
                    result = blend_texture(result, mask_shams, warped_sham, brightness=conf_brightness)
                
                col_idx = idx % 2
                with cols[col_idx]:
                    st.image(result, caption=f"{view_name.title()} View", use_container_width=True)
                    
                    pil_img = Image.fromarray(result)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=90)
                    
                    st.download_button(
                        label=f"Download {view_name.title()}",
                        data=buf.getvalue(),
                        file_name=f"render_{view_name}.jpg",
                        mime="image/jpeg"
                    )
            
            except Exception as e:
                st.error(f"Error processing {view_name}: {str(e)}")

else:
    st.info("Upload a pattern on the left sidebar to begin.")