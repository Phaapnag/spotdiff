import os
import time
from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import gradio as gr

# ====== 參數 ======
OUTPUTDIR = "output"
FPS = 24
QUIZSECONDS = 10
ANSWERSECONDS = 2
MAX_DISPLAY = 1024
MAX_VIDEO_WIDTH = 540  # 限制影片寬度 540p，避免 OOM

# ====== Helper ======
def ensure_uint8_array(data):
    """確保資料是乾淨的 uint8 numpy array"""
    while isinstance(data, list):
        if len(data) == 0: return None
        data = data[0]
    arr = np.array(data)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == 'f' and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr

def resize_for_display(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_DISPLAY: return img
    scale = MAX_DISPLAY / float(longest)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

def load_and_align_images(base_img: Image.Image, variant_img: Image.Image):
    img1 = base_img.convert("RGB")
    img2 = variant_img.convert("RGB")
    w, h = img1.size
    if img2.size != (w, h):
        img2 = img2.resize((w, h), Image.LANCZOS)
    return img1, img2

def draw_circles_on_image(img: Image.Image, points: list) -> Image.Image:
    if img is None: return None
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    for p in points:
        # points 支援 (x, y, r, t)
        if len(p) == 4:
            x, y, r, t = p
            cv2.circle(bgr, (int(x), int(y)), int(r), (0, 0, 255), int(t))
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def draw_text_opencv(imgbgr: np.ndarray, text: str):
    caption_font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    h, w = imgbgr.shape[:2]
    barheight = 60
    cv2.rectangle(imgbgr, (0, 0), (w, barheight), (0, 0, 0), -1)
    (textw, texth), _ = cv2.getTextSize(text, caption_font, scale, thickness)
    x = (w - textw) // 2
    y = (barheight + texth) // 2
    cv2.putText(imgbgr, text, (x, y), caption_font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return imgbgr

def make_video_with_opencv_frames(img1, img2, img2_marked, outpath):
    # ★ 極限省記憶體版：直接用 FFMPEG Writer 串流寫入，不囤積 frames list
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    img1bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img2markedbgr = cv2.cvtColor(np.array(img2_marked), cv2.COLOR_RGB2BGR)
    
    h, w = img1bgr.shape[:2]
    
    # 強制縮圖到 540p 以內
    if w > MAX_VIDEO_WIDTH:
        scale = MAX_VIDEO_WIDTH / w
        new_w, new_h = MAX_VIDEO_WIDTH, int(h * scale)
        if new_w % 2 != 0: new_w -= 1
        if new_h % 2 != 0: new_h -= 1
        img1bgr = cv2.resize(img1bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img2bgr = cv2.resize(img2bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img2markedbgr = cv2.resize(img2markedbgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    fullheight = h * 2
    fps = FPS
    
    writer = FFMPEG_VideoWriter(outpath, (w, fullheight), fps)
    
    # Quiz 部分
    for i in range(QUIZSECONDS * fps):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2bgr
        
        remaining = QUIZSECONDS - i / fps
        frame = draw_text_opencv(frame, f"找出 5 個不同！剩餘 {remaining:.0f} 秒")
        writer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    # Answer 部分
    for _ in range(ANSWERSECONDS * fps):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2markedbgr
        frame = draw_text_opencv(frame, "答案在下面！")
        writer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

# ====== Gradio Functions ======

def step1_align(base_file, variant_file):
    os.makedirs(OUTPUTDIR, exist_ok=True)
    if base_file is None or variant_file is None:
        return None, None, None, None, None

    img1, img2 = load_and_align_images(
        Image.fromarray(ensure_uint8_array(base_file)), 
        Image.fromarray(ensure_uint8_array(variant_file))
    )
    
    # 存硬碟
    img1.save(os.path.join(OUTPUTDIR, "base_aligned.jpg"))
    img2.save(os.path.join(OUTPUTDIR, "variant_aligned.jpg"))
    
    # 縮圖顯示
    img1_disp = resize_for_display(img1)
    img2_disp = resize_for_display(img2)
    
    return img1_disp, img2_disp, img2_disp, np.array(img1), np.array(img2)

def on_click_variant(variant_original, evt: gr.SelectData, radius, thickness, points):
    if variant_original is None: return None, points
    
    x, y = evt.index
    points = list(points or [])
    
    if len(points) < 5:
        points.append((x, y, radius, thickness))
    
    variant_original_img = Image.fromarray(ensure_uint8_array(variant_original))
    marked = draw_circles_on_image(variant_original_img, points)
    return np.array(marked), points

def reset_points(variant_original):
    if variant_original is None: return None, []
    return variant_original, []

def undo_last_point(variant_original, points):
    points = list(points or [])
    if points: points.pop()
    
    if variant_original is None: return None, points
    
    variant_original_img = Image.fromarray(ensure_uint8_array(variant_original))
    marked = draw_circles_on_image(variant_original_img, points)
    return np.array(marked), points

def get_images_from_state_or_disk(base_full_state, variant_full_state):
    """共用函數：嘗試從硬碟或 State 取得原圖"""
    base_path = os.path.join(OUTPUTDIR, "base_aligned.jpg")
    variant_path = os.path.join(OUTPUTDIR, "variant_aligned.jpg")
    
    img1, img2 = None, None
    if os.path.exists(base_path) and os.path.exists(variant_path):
        img1 = Image.open(base_path).convert("RGB")
        img2 = Image.open(variant_path).convert("RGB")
    else:
        if base_full_state is None: return None, None
        img1 = Image.fromarray(ensure_uint8_array(base_full_state))
        img2 = Image.fromarray(ensure_uint8_array(variant_full_state))
    return img1, img2

def preview_final_frames(base_full_state, variant_full_state, points):
    """預覽合成圖"""
    img1, img2 = get_images_from_state_or_disk(base_full_state, variant_full_state)
    if img1 is None: raise gr.Error("請回到步驟 1 按『開始』。")
    if not points: raise gr.Error("請先標記紅圈。")

    img2_marked = draw_circles_on_image(img2, points)
    
    # 簡單合成預覽
    w, h = img1.size
    fullheig
