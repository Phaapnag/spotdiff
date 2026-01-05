import gradio as gr

# ... 你原有的所有 import 和函數 (loadandalignimages, manualmarkpoints, makevideowithopencvframes 等) 保持不變 ...

def process_images(base_file, variant_file):
    os.makedirs("output", exist_ok=True)
    
    # 存檔
    base_path = "output/base.jpg"
    variant_path = "output/variant.jpg"
    base_file.save(base_path)
    variant_file.save(variant_path)
    
    # 載入對齊
    img1, img2 = loadandalignimages(base_path, variant_path)
    
    # 自動存 aligned 圖給 UI 用
    img1.save("output/base_aligned.jpg")
    img2.save("output/variant_aligned.jpg")
    
    return "output/base_aligned.jpg", "output/variant_aligned.jpg"

def generate_video(marked_variant_file):
    img1 = Image.open("output/base_aligned.jpg")
    img2_marked = Image.open(marked_variant_file)
    video_path = "output/output.mp4"
    makevideowithopencvframes(img1, img2_marked, img2_marked, video_path)
    return video_path

# Gradio UI
iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.Image(type="pil", label="上傳基準圖 (base.png)"),
        gr.Image(type="pil", label="上傳變體圖 (variant.png)")
    ],
    outputs=[
        gr.Image(label="基準圖 (已對齊)"),
        gr.Image(label="變體圖 (已對齊，準備畫紅圈)")
    ],
    title="找不同 Shorts 生成器",
    description="1. 上傳兩張圖 → 2. 下載變體圖，在本地用滑鼠畫5個紅圈 → 3. 上傳標記後圖 → 4. 生成12秒影片"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
