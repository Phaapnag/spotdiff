import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import moviepy.editor as mpy
import gradio as gr

# ====== 原有參數 ======
OUTPUTDIR = "output"
FPS = 24
QUIZSECONDS = 10
ANSWERSECONDS = 2
TITLE = "找不同 Shorts 生成器"


# ====== 共用函數 ======
def load_and_align_images(base_img: Image.Image, variant_img: Image.Image):
    """將兩張 PIL Image 對齊到相同尺寸。"""
    img1 = base_img.convert("RGB")
    img2 = variant_img.convert("RGB")
    w, h = img1.size
    if img2.size != (w, h):
        img2 = img2.resize((w, h), Image.LANCZOS)
    return img1, img2


def draw_circles_on_image(
    img: Image.Image,
    points: List[Tuple[int, int]],
    radius: int,
    thickness: int,
    color=(255, 0, 0),
) -> Image.Image:
    """在 PIL 圖片上畫紅圈，回傳新的 PIL 圖片。"""
    if img is None:
        return None
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for x, y in points:
        cv2.circle(bgr, (int(x), int(y)), int(radius), (0, 0, 255), int(thickness))
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


def make_video_with_opencv_frames(
    img1: Image.Image, img2: Image.Image, img2_marked: Image.Image, outpath: str
):
    totalquizframes = QUIZSECONDS * FPS
    totalanswerframes = ANSWERSECONDS * FPS

    img1bgr = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    img2bgr = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    img2markedbgr = cv2.cvtColor(np.array(img2_marked), cv2.COLOR_RGB2BGR)

    h, w = img1bgr.shape[:2]
    width = w
    fullheight = h * 2
    frames = []

    # Quiz 部分
    for i in range(totalquizframes):
        frame = np.zeros((fullheight, width, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2bgr

        remaining = QUIZSECONDS - i / FPS
        text = f"找出 5 個不同！剩餘 {remaining:.0f} 秒"
        frame = draw_text_opencv(frame, text)

        framesmall = cv2.resize(frame, (width // 2, fullheight // 2), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(framesmall, cv2.COLOR_BGR2RGB))

    # Answer 部分
    for _ in range(totalanswerframes):
        frame = np.zeros((fullheight, width, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2markedbgr

        frame = draw_text_opencv(frame, "答案在下面！")

        framesmall = cv2.resize(frame, (width // 2, fullheight // 2), interpolation=cv2.INTER_LINEAR)
        frames.append(cv2.cvtColor(framesmall, cv2.COLOR_BGR2RGB))

    clip = mpy.ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(outpath, codec="libx264", audio=False)


# ====== Gradio 相關函數 ======
MAX_DISPLAY = 1024  # UI 顯示時的最大邊長（像素）

def resize_for_display(img: Image.Image) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= MAX_DISPLAY:
        return img  # 已經不大，直接用
    scale = MAX_DISPLAY / float(longest)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


def step1_align(base_file, variant_file):
    """Step1: 上傳兩張圖並對齊，回傳給 UI 用的【縮細版】 base / variant。"""
    os.makedirs(OUTPUTDIR, exist_ok=True)
    if base_file is None or variant_file is None:
        return None, None

    base_img = Image.fromarray(base_file) if isinstance(base_file, np.ndarray) else base_file
    variant_img = (
        Image.fromarray(variant_file) if isinstance(variant_file, np.ndarray) else variant_file
    )

    # 先對齊原始尺寸
    img1, img2 = load_and_align_images(base_img, variant_img)

    # 存一份「原圖對齊」給之後做影片用（如果你現在影片也是用 base_aligned / variant_aligned）
    base_aligned = os.path.join(OUTPUTDIR, "base_aligned.jpg")
    variant_aligned = os.path.join(OUTPUTDIR, "variant_aligned.jpg")
    img1.save(base_aligned)
    img2.save(variant_aligned)

    # 再做一份「縮細版」給 UI 顯示，減少每次畫圈傳輸量
    img1_disp = resize_for_display(img1)
    img2_disp = resize_for_display(img2)

    return img1_disp, img2_disp



def on_click_variant(img, evt: gr.SelectData, radius, thickness, points):
    """在變體圖上點擊時，新增一個紅圈並回傳新的圖與 points。"""
    if img is None:
        return None, points

    # evt.index = (x, y)
    x, y = evt.index
    points = list(points or [])

    # 限制最多 5 個點
    if len(points) >= 5:
        return draw_circles_on_image(Image.fromarray(img), points, radius, thickness), points

    points.append((x, y))
    marked = draw_circles_on_image(Image.fromarray(img), points, radius, thickness)
    return np.array(marked), points


def reset_points(img):
    """重設紅圈。"""
    return img, []

def undo_last_point(img, points, radius, thickness):
    """刪除最後一個紅圈並重畫。"""
    points = list(points or [])
    if not points:
        return img, points  # 沒有點就不變

    points.pop()  # 刪掉最後一個
    if img is None:
        return img, points

    # 重新在原圖上畫剩下的點
    pil_img = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    marked = draw_circles_on_image(pil_img, points, radius, thickness)
    return np.array(marked), points


def step2_make_video(points, radius, thickness):
    """Step2: 用 base_aligned + variant_aligned + points 生成影片。"""
    if not points:
        raise gr.Error("請先在變體圖上點擊，標記至少 1 個紅圈（最多 5 個）。")

    base_path = os.path.join(OUTPUTDIR, "base_aligned.jpg")
    variant_path = os.path.join(OUTPUTDIR, "variant_aligned.jpg")
    if not (os.path.exists(base_path) and os.path.exists(variant_path)):
        raise gr.Error("請先完成步驟 1 上傳並對齊圖片。")

    img1 = Image.open(base_path).convert("RGB")
    img2 = Image.open(variant_path).convert("RGB")

    img2_marked = draw_circles_on_image(img2, points, radius, thickness)
    marked_path = os.path.join(OUTPUTDIR, "variant_marked.jpg")
    img2_marked.save(marked_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f"spotdiff_{timestamp}.mp4"
    video_path = os.path.join(OUTPUTDIR, video_filename)

    make_video_with_opencv_frames(img1, img2, img2_marked, video_path)
    return video_path


# ====== 建立 Gradio 介面 ======
with gr.Blocks(title="找不同 Shorts 生成器") as demo:
    gr.Markdown(
        "## 🔍 找不同 Shorts 生成器\n"
        "1️⃣ 上傳兩張圖 → 2️⃣ 在下方變體圖點 5 個紅圈（可調圈圈大小 & 粗幼）→ "
        "3️⃣ 生成 12 秒 YouTube Shorts MP4！"
    )

    # State 用來存 points
    points_state = gr.State([])

    with gr.Tab("步驟 1：上傳 & 對齊"):
        with gr.Row():
            base_input = gr.Image(label="📸 上傳基準圖 (base)", type="pil")
            variant_input = gr.Image(label="📸 上傳變體圖 (variant)", type="pil")
        align_button = gr.Button("✅ 對齊並顯示")

        with gr.Row():
            base_show = gr.Image(
                label="基準圖 (已對齊)",
                height=600,
            )
            variant_show = gr.Image(
                label="變體圖 (點擊畫紅圈)", 
                interactive=True, 
                height=600,
            )

        radius_slider = gr.Slider(
            minimum=10,
            maximum=300,
            value=40,
            step=2,
            label="🔴 紅圈半徑 (越大圈越大)",
        )
        thickness_slider = gr.Slider(
            minimum=2,
            maximum=20,
            value=6,
            step=1,
            label="🖊 線條粗幼",
        )
        reset_button = gr.Button("♻️ 重設所有紅圈")
        undo_button = gr.Button("↩️ Undo 上一個紅圈")   # ★ 新增


        # Step1 對齊
        align_button.click(
            fn=step1_align,
            inputs=[base_input, variant_input],
            outputs=[base_show, variant_show],
        )

        # 點擊變體圖時畫圈
        variant_show.select(
            fn=on_click_variant,
            inputs=[variant_show, radius_slider, thickness_slider, points_state],
            outputs=[variant_show, points_state],
        )

        # 重設紅圈
        reset_button.click(
            fn=reset_points,
            inputs=[variant_show],
            outputs=[variant_show, points_state],
        )

        # Undo 最後一個紅圈
        undo_button.click(
            fn=undo_last_point,
            inputs=[variant_show, points_state, radius_slider, thickness_slider],
            outputs=[variant_show, points_state],
        )


    with gr.Tab("步驟 2：生成影片"):
        gr.Markdown("確認紅圈後，按下方按鈕生成 12 秒影片。")
        make_video_button = gr.Button("🎥 生成 12 秒 MP4")
        video_output = gr.Video(label="輸出影片", interactive=False)

        make_video_button.click(
            fn=step2_make_video,
            inputs=[points_state, radius_slider, thickness_slider],
            outputs=video_output,
        )


if __name__ == "__main__":
    os.makedirs(OUTPUTDIR, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
