import gradio as gr
from main import process_images, generate_video  # 匯入你的函數

demo = gr.Interface(...)  # copy 上面的 iface 定義

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
