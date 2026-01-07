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

# ====== Helper Functions ======
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
    # FFMPEG Writer 串流寫入 (省記憶體)
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

    img1bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img2markedbgr = cv2.cvtColor(np.array(img2_marked), cv2.COLOR_RGB2BGR)
    
    h, w = img1bgr.shape[:2]
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
    
    for i in range(QUIZSECONDS * fps):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2bgr
        remaining = QUIZSECONDS - i / fps
        frame = draw_text_opencv(frame, f"找出 5 個不同！剩餘 {remaining:.0f} 秒")
        writer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    for _ in range(ANSWERSECONDS * fps):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2markedbgr
        frame = draw_text_opencv(frame, "答案在下面！")
        writer.write_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    writer.close()

# ====== Gradio Logic ======

def step1_align(base_file, variant_file):
    os.makedirs(OUTPUTDIR, exist_ok=True)
    if base_file is None or variant_file is None:
        return None, None, None, None, None

    img1, img2 = load_and_align_images(
        Image.fromarray(ensure_uint8_array(base_file)), 
        Image.fromarray(ensure_uint8_array(variant_file))
    )
    
    img1.save(os.path.join(OUTPUTDIR, "base_aligned.jpg"))
    img2.save(os.path.join(OUTPUTDIR, "variant_aligned.jpg"))
    
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
    img1, img2 = get_images_from_state_or_disk(base_full_state, variant_full_state)
    if img1 is None: raise gr.Error("請回到步驟 1 按『開始』。")
    # ★ 允許空 points，不報錯
    
    img2_marked = draw_circles_on_image(img2, points)
    w, h = img1.size
    fullheight = h * 2
    
    canvas1 = Image.new("RGB", (w, fullheight))
    canvas1.paste(img1, (0, 0))
    canvas1.paste(img2, (0, h))
    
    canvas2 = Image.new("RGB", (w, fullheight))
    canvas2.paste(img1, (0, 0))
    canvas2.paste(img2_marked, (0, h))
    
    return resize_for_display(canvas1), resize_for_display(canvas2)

def step2_make_video(base_full_state, variant_full_state, points):
    img1, img2 = get_images_from_state_or_disk(base_full_state, variant_full_state)
    if img1 is None: raise gr.Error("請回到步驟 1 按『開始』。")
    # ★ 允許空 points，不報錯
    
    img2_marked = draw_circles_on_image(img2, points)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(OUTPUTDIR, f"spotdiff_{timestamp}.mp4")
    
    make_video_with_opencv_frames(img1, img2, img2_marked, video_path)
    return video_path

# ====== UI ======
with gr.Blocks(title="找不同 Shorts 生成器", css=".img-display { object-fit: contain; }") as demo:
    points_state = gr.State([])
    variant_original_state = gr.State(None)
    base_full_state = gr.State(None)
    variant_full_state = gr.State(None)
    
    with gr.Tab("步驟 1：上傳 & 對齊"):
        with gr.Row():
            base_input = gr.Image(label="基準圖", type="numpy")
            variant_input = gr.Image(label="變體圖", type="numpy")
        align_btn = gr.Button("✅ 開始（上傳 & 對齊）")
        
        with gr.Row():
            base_show = gr.Image(label="基準圖 (已對齊)", height=600, elem_classes="img-display")
            variant_show = gr.Image(label="變體圖 (點擊畫紅圈)", height=600, interactive=True, elem_classes="img-display")
            
        with gr.Row():
            radius_slider = gr.Slider(10, 300, 40, 2, label="🔴 紅圈半徑")
            thickness_slider = gr.Slider(2, 20, 6, 1, label="🖊 線條粗幼")
            
        with gr.Row():
            reset_btn = gr.Button("♻️ 重設")
            undo_btn = gr.Button("↩️ Undo")
            
        align_btn.click(step1_align, [base_input, variant_input], 
                        [base_show, variant_show, variant_original_state, base_full_state, variant_full_state])
        
        variant_show.select(on_click_variant, 
                            [variant_original_state, radius_slider, thickness_slider, points_state], 
                            [variant_show, points_state])
                            
        reset_btn.click(reset_points, [variant_original_state], [variant_show, points_state])
        undo_btn.click(undo_last_point, [variant_original_state, points_state], [variant_show, points_state])
        
    with gr.Tab("步驟 2：生成影片"):
        preview_btn = gr.Button("🔍 預覽合成圖")
        with gr.Row():
            # ★ 高度改為 300，縮小預覽視窗
            preview_quiz = gr.Image(label="Quiz 畫面 (無圈)", height=300, elem_classes="img-display")
            preview_answer = gr.Image(label="Answer 畫面 (有圈)", height=300, elem_classes="img-display")
            
        make_video_btn = gr.Button("🎥 生成 12 秒 MP4")
        video_out = gr.Video(label="輸出影片")
        
        preview_btn.click(preview_final_frames, 
                          [base_full_state, variant_full_state, points_state],
                          [preview_quiz, preview_answer])
                          
        make_video_btn.click(step2_make_video, 
                             [base_full_state, variant_full_state, points_state], 
                             video_out)

if __name__ == "__main__":
    os.makedirs(OUTPUTDIR, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
