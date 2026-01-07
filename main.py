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
    
    # ★ 改：輸出影片最大寬度 720（YouTube Shorts 夠用），減少記憶體
    MAX_VIDEO_WIDTH = 720
    if w > MAX_VIDEO_WIDTH:
        scale = MAX_VIDEO_WIDTH / w
        new_w = MAX_VIDEO_WIDTH
        new_h = int(h * scale)
        img1bgr = cv2.resize(img1bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img2bgr = cv2.resize(img2bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img2markedbgr = cv2.resize(img2markedbgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        w, h = new_w, new_h

    fullheight = h * 2
    frames = []

    # Quiz 部分
    for i in range(totalquizframes):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2bgr

        remaining = QUIZSECONDS - i / FPS
        text = f"找出 5 個不同！剩餘 {remaining:.0f} 秒"
        frame = draw_text_opencv(frame, text)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Answer 部分
    for _ in range(totalanswerframes):
        frame = np.zeros((fullheight, w, 3), dtype=np.uint8)
        frame[0:h, :, :] = img1bgr
        frame[h:fullheight, :, :] = img2markedbgr
        frame = draw_text_opencv(frame, "答案在下面！")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    clip = mpy.ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(outpath, codec="libx264", audio=False, preset="ultrafast")  # ★ 加 preset 加速



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


# 多加一個 state 存高清圖
base_full_state = gr.State(None)
variant_full_state = gr.State(None)

def step1_align(base_file, variant_file):
    """Step1: 上傳兩張圖並對齊，回傳給 UI 用的【縮細版】 base / variant，並保存原始圖到 state。"""
    os.makedirs(OUTPUTDIR, exist_ok=True)
    if base_file is None or variant_file is None:
        return None, None, None, None, None

    base_img = Image.fromarray(base_file) if isinstance(base_file, np.ndarray) else base_file
    variant_img = (
        Image.fromarray(variant_file) if isinstance(variant_file, np.ndarray) else variant_file
    )

    # 先對齊原始尺寸（高清）
    img1, img2 = load_and_align_images(base_img, variant_img)

    # 存一份「原圖對齊」給之後做影片用（可選，備份）
    base_aligned = os.path.join(OUTPUTDIR, "base_aligned.jpg")
    variant_aligned = os.path.join(OUTPUTDIR, "variant_aligned.jpg")
    img1.save(base_aligned)
    img2.save(variant_aligned)

    # 轉成 numpy，放在 state 裡（高清版本）
    base_np = np.array(img1)
    variant_np = np.array(img2)

    # 再做一份「縮細版」給 UI 顯示，減少每次畫圈傳輸量
    img1_disp = resize_for_display(img1)
    img2_disp = resize_for_display(img2)

    # 回傳：顯示用 base、顯示用 variant、原始顯示用 variant、高清 base、高清 variant
    return img1_disp, img2_disp, img2_disp, base_np, variant_np
    





def on_click_variant(variant_original, evt: gr.SelectData, radius, thickness, points):
    """在變體圖上點擊時，新增一個紅圈並回傳新的圖與 points。"""
    if variant_original is None:
        return None, points

    x, y = evt.index
    points = list(points or [])

    # 限制最多 5 個點
    if len(points) >= 5:
        marked = draw_circles_on_image(variant_original, points, radius, thickness)
        return np.array(marked), points

    points.append((x, y))
    marked = draw_circles_on_image(variant_original, points, radius, thickness)
    return np.array(marked), points



def reset_points(variant_original):
    """重設紅圈：回到原始變體圖，清空 points。"""
    if variant_original is None:
        return None, []
    return np.array(variant_original), []


def undo_last_point(variant_original, points, radius, thickness):
    """刪除最後一個紅圈並重畫。"""
    points = list(points or [])
    if not points or variant_original is None:
        return (np.array(variant_original) if variant_original is not None else None), points

    points.pop()
    marked = draw_circles_on_image(variant_original, points, radius, thickness)
    return np.array(marked), points

def preview_final_frames(points, radius, thickness):
    """生成最終兩張合成圖：上 = base+variant（無圈），下 = base+variant（有圈）。"""
    if not points:
        raise gr.Error("請先在步驟 1 標記紅圈。")

    base_path = os.path.join(OUTPUTDIR, "base_aligned.jpg")
    variant_path = os.path.join(OUTPUTDIR, "variant_aligned.jpg")
    if not (os.path.exists(base_path) and os.path.exists(variant_path)):
        raise gr.Error("請先完成步驟 1 上傳並對齊圖片。")

    img1 = Image.open(base_path).convert("RGB")
    img2 = Image.open(variant_path).convert("RGB")
    img2_marked = draw_circles_on_image(img2, points, radius, thickness)

    w, h = img1.size
    fullheight = h * 2

    # 合成圖 1：base 上 + variant 下（無圈）
    canvas1 = np.zeros((fullheight, w, 3), dtype=np.uint8)
    canvas1[0:h, :, :] = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    canvas1[h:fullheight, :, :] = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR)
    preview1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB)

    # 合成圖 2：base 上 + variant 下（有圈）
    canvas2 = np.zeros((fullheight, w, 3), dtype=np.uint8)
    canvas2[0:h, :, :] = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2BGR)
    canvas2[h:fullheight, :, :] = cv2.cvtColor(np.array(img2_marked), cv2.COLOR_RGB2BGR)
    preview2 = cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB)

    return Image.fromarray(preview1), Image.fromarray(preview2)


def step2_make_video(base_full, variant_full, points, radius, thickness):
    """Step2: 用 state 裡的高清 base / variant + points 生成影片。"""
    if base_full is None or variant_full is None:
        raise gr.Error("請先在步驟 1 上傳並對齊圖片（按一次『開始（上傳 & 對齊）』）。")

    if not points:
        raise gr.Error("請先在變體圖上點擊，標記至少 1 個紅圈（最多 5 個）。")

    import numpy as np  # 上面已經有

    # 如果是 list，就取第一個元素
    if isinstance(base_full, list):
        base_full = base_full[0]
    if isinstance(variant_full, list):
        variant_full = variant_full[0]

    img1 = Image.fromarray(np.array(base_full))
    img2 = Image.fromarray(np.array(variant_full))


    # 畫上紅圈，得到標記後的變體圖
    img2_marked = draw_circles_on_image(img2, points, radius, thickness)
    marked_path = os.path.join(OUTPUTDIR, "variant_marked.jpg")
    img2_marked.save(marked_path)

    # 生成影片
    os.makedirs(OUTPUTDIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_filename = f"spotdiff_{timestamp}.mp4"
    video_path = os.path.join(OUTPUTDIR, video_filename)

    make_video_with_opencv_frames(img1, img2, img2_marked, video_path)
    return video_path



# ====== 建立 Gradio 介面 ======
with gr.Blocks(title="找不同 Shorts 生成器") as demo:
    gr.Markdown(
        "## 🔍 找不同 Shorts 生成器\n"
        "1️⃣ 上傳兩張圖 → 2️⃣ 在下方變體圖點 5 個紅圈（可調圈圈大小 & 粗幼, 中途不要再按「開始」）→ "
        "3️⃣ 生成 12 秒 YouTube Shorts MP4！"
    )

    # State 用來存 points
    points_state = gr.State([])
    variant_original_state = gr.State(None)  # ★ 新增：保存「未畫圈」的變體圖
    base_full_state = gr.State(None)      # 存高清的 base
    variant_full_state = gr.State(None)   # 存高清的 variant

    
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
        outputs=[base_show, variant_show, variant_original_state, base_full_state, variant_full_state],
        )


        # 點擊變體圖時畫圈
        variant_show.select(
            fn=on_click_variant,
            inputs=[variant_original_state, radius_slider, thickness_slider, points_state],
            outputs=[variant_show, points_state],
        )


        # 重設紅圈
        reset_button.click(
            fn=reset_points,
            inputs=[variant_original_state],
            outputs=[variant_show, points_state],
        )


        # Undo 最後一個紅圈
        undo_button.click(
            fn=undo_last_point,
            inputs=[variant_original_state, points_state, radius_slider, thickness_slider],
            outputs=[variant_show, points_state],
        )



    with gr.Tab("步驟 2：生成影片"):
        gr.Markdown("確認紅圈後，先預覽合成效果，再生成 12 秒影片。")
        
        preview_button = gr.Button("🔍 預覽合成圖（影片前 10 秒 vs 後 2 秒）")
        with gr.Row():
            preview_quiz = gr.Image(label="📺 Quiz 畫面：base + variant（無圈）")
            preview_answer = gr.Image(label="📺 Answer 畫面：base + variant（有圈）")
        
        make_video_button = gr.Button("🎥 確認無誤，生成 12 秒 MP4")
        video_output = gr.Video(label="輸出影片", interactive=False)

        # 預覽
        preview_button.click(
            fn=preview_final_frames,
            inputs=[points_state, radius_slider, thickness_slider],
            outputs=[preview_quiz, preview_answer],
        )

        # 生成影片
        make_video_button.click(
            fn=step2_make_video,
            inputs=[points_state, radius_slider, thickness_slider],
            outputs=video_output,
        )



if __name__ == "__main__":
    os.makedirs(OUTPUTDIR, exist_ok=True)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
