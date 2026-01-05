import os
import time
import PIL
from PIL import Image
import cv2
import numpy as np
import moviepy.editor as mpy
import gradio as gr

# 原有設定
IMGBASEPATH = "base.png"
IMGVARIANTPATH = "variant.png"
OUTPUTDIR = "output"
FPS = 24
QUIZSECONDS = 10
ANSWERSECONDS = 2
TITLE = "Spot the 5 Differences!"

# 原有函數 (完全不變)
def loadandalignimages(basepath: str, variantpath: str):
    base = Image.open(basepath).convert('RGB')
    variant = Image.open(variantpath).convert('RGB')
    w, h = base.size
    if variant.size != (w, h):
        variant = variant.resize((w, h), Image.LANCZOS)
    return base, variant

def manualmarkpoints(img1: Image.Image, img2: Image.Image, savepath: str, maxpoints: int = 5):
    img1bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    h, w = img1bgr.shape[:2]
    fullheight = h * 2
    canvas = np.zeros((fullheight, w, 3), dtype=np.uint8)
    canvas[0:h, :, :] = img1bgr
    canvas[h:fullheight, :, :] = img2bgr
    
    points = []
    radius = 40
    
    def redrawcanvas():
        nonlocal canvas
        canvas = np.zeros((fullheight, w, 3), dtype=np.uint8)
        canvas[0:h, :, :] = img1bgr
        canvas[h:fullheight, :, :] = img2bgr
        for x, y in points:
            cv2.circle(canvas, (x, y), radius, (0, 0, 255), 4)
    
    def onmouse(event, x, y, flags, param):
        nonlocal canvas, points
        if event == cv2.EVENT_LBUTTONDOWN and y >= h and len(points) < maxpoints:
            points.append((x, y))
            redrawcanvas()
            cv2.imshow('Click 5 differences on variant', canvas)
    
    print(f"{savepath} - maxpoints={maxpoints}")
    print("  - SPACE=+, - =-, u=undo, ESC=done")
    
    cv2.namedWindow('Click 5 differences on variant', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Click 5 differences on variant', w, fullheight)
    redrawcanvas()
    cv2.imshow('Click 5 differences on variant', canvas)
    cv2.setMouseCallback('Click 5 differences on variant', onmouse)
    
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord(' ') or key == ord('+'):
            radius = min(radius + 5, 200)
            redrawcanvas()
            cv2.imshow('Click 5 differences on variant', canvas)
        elif key == ord('-') or key == ord('='):
            radius = max(radius - 5, 5)
            redrawcanvas()
            cv2.imshow('Click 5 differences on variant', canvas)
        elif key == ord('u'):
            if points:
                points.pop()
                redrawcanvas()
                cv2.imshow('Click 5 differences on variant', canvas)
    
    cv2.destroyAllWindows()
    
    img2mark = img2bgr.copy()
    for x, y in points:
        yy = y - h
        cv2.circle(img2mark, (x, yy), radius, (0, 0, 255), 4)
    cv2.imwrite(savepath, img2mark)
    print(f"Saved: {savepath}")

def drawtextopencv(imgbgr: np.ndarray, text: str):
    caption_font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 4.0
    thickness = 6
    h, w = imgbgr.shape[:2]
    barheight = 260
    cv2.rectangle(imgbgr, (0, 0), (w, barheight), (0, 0, 0), thickness - 1)
    
    (textw, texth), _ = cv2.getTextSize(text, caption_font, scale, thickness)
    x = (w - textw) // 2
    y = (barheight - texth) // 2
    cv2.putText(imgbgr, text, (x, y), caption_font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return imgbgr

def makevideowithopencvframes(img1: Image.Image, img2: Image.Image, img2marked: Image.Image, outpath: str):
    totalquizframes = QUIZSECONDS * FPS
    totalanswerframes = ANSWERSECONDS * FPS
    
    img1bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img2markedbgr = cv2.cvtColor(np.array(img2marked), cv2.COLOR_RGB2BGR)
    
    h, w = img1bgr.shape[:2]
    width = w
    fullheight = h * 2
    frames = []
    
    # Quiz part
    for i in range(totalquizframes):
        frame = np.zeros((fullheight, width, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2bgr
        
        remaining = QUIZSECONDS - i / FPS
        text = f"Spot 5 differences! remaining={remaining:.0f}s"
        frame = drawtextopencv(frame, text)
        
        framesmall = cv2.resize(frame, (width//2, fullheight//2), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(framesmall, cv2.COLOR_BGR2RGB))
    
    # Answer part
    for i in range(totalanswerframes):
        frame = np.zeros((fullheight, width, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2markedbgr
        
        frame = drawtextopencv(frame, "Answers!")
        
        framesmall = cv2.resize(frame, (width//2, fullheight//2), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(framesmall, cv2.COLOR_BGR2RGB))
    
    clip = mpy.ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(outpath, codec='libx264', audio=False)

# ========== 新增：Gradio Web UI 功能 ==========
def process_images(base_file, variant_file):
    os.makedirs(OUTPUTDIR, exist_ok=True)
    
    base_path = f"{OUTPUTDIR}/base.jpg"
    variant_path = f"{OUTPUTDIR}/variant.jpg"
    base_file.save(base_path)
    variant_file.save(variant_path)
    
    img1, img2 = loadandalignimages(base_path, variant_path)
    
    base_aligned = f"{OUTPUTDIR}/base_aligned.jpg"
    variant_aligned = f"{OUTPUTDIR}/variant_aligned.jpg"
    img1.save(base_aligned)
    img2.save(variant_aligned)
    
    return base_aligned, variant_aligned

def generate_video(marked_file):
    img1 = Image.open(f"{OUTPUTDIR}/base_aligned.jpg")
    img2_marked = Image.open(marked_file)
    video_path = f"{OUTPUTDIR}/output.mp4"
    makevideowithopencvframes(img1, img2_marked, img2_marked, video_path)
    return video_path

# Gradio 介面
iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="pil", label="📸 上傳基準圖 (base)"),
        gr.Image(type="pil", label="📸 上傳變體圖 (variant)")
    ],
    outputs=[
        gr.Image(label="✅ 基準圖 (已對齊)"),
        gr.Image(label="🎯 變體圖 (已對齊，下載後畫5個紅圈)")
    ],
    title="🔍 找不同 Shorts 生成器",
    description="""1️⃣ 上傳兩張圖 → 2️⃣ 下載「變體圖」，**本地用滑鼠畫5個紅圈** → 3️⃣ 上傳標記後圖 → 4️⃣ 生成12秒MP4！"""
)

# 第二個介面：生成影片
video_iface = gr.Interface(
    fn=generate_video,
    inputs=gr.Image(type="pil", label="📤 上傳已畫紅圈的變體圖"),
    outputs=gr.Image(type="filepath", label="🎥 下載12秒 YouTube Shorts MP4"),
    title="生成影片"
)

demo = gr.TabbedInterface([iface, video_iface], ["步驟1: 上傳&對齊", "步驟2: 生成影片"])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )
