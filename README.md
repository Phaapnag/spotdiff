# 🎯 Spot the Difference Video Generator

An interactive web app that generates "Spot the Difference" challenge videos for YouTube Shorts, TikTok, and social media. Built with Gradio and Python.

## ✨ Features

- **Interactive Web Interface**: Upload two images and mark differences with clicks
- **Smart Image Alignment**: Automatically aligns and resizes images to match dimensions
- **Visual Circle Marking**: Click to place red circles on differences with adjustable size and thickness
- **Real-time Preview**: See quiz and answer frames before generating video
- **Auto Video Generation**: Creates 12-second MP4 videos (10s quiz + 2s answer)
- **Memory Efficient**: Optimized for handling large images without OOM errors
- **Gradio UI**: User-friendly interface with no coding required

## 🎬 Video Output Format

- **Duration**: 12 seconds total
  - 10 seconds: Quiz phase with countdown timer
  - 2 seconds: Answer reveal with marked differences
- **Layout**: Vertical split screen (perfect for Shorts/TikTok)
  - Top half: Base image
  - Bottom half: Variant image (with/without circles)
- **Resolution**: Auto-scaled to 540p width for optimal performance
- **FPS**: 24 frames per second

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- FFmpeg (for video encoding)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Phaapnag/spotdiff.git
cd spotdiff
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python main.py
```

4. Open your browser:
```
http://localhost:7860
```

## 📖 How to Use

### Step 1: Upload & Align

1. Upload your **base image** (original)
2. Upload your **variant image** (with differences)
3. Click **"✅ 開始（上傳 & 對齊）"** to align images
4. Click on the variant image to mark up to 5 differences
   - Adjust **red circle radius** with the slider
   - Adjust **line thickness** for visibility
5. Use **"↩️ Undo"** to remove last circle
6. Use **"♻️ 重設"** to clear all circles

### Step 2: Generate Video

1. Click **"🔍 預覽合成圖"** to preview final frames
   - Left: Quiz frame (no circles)
   - Right: Answer frame (with circles)
2. Click **"🎥 生成 12 秒 MP4"** to create the video
3. Download your video from the output section

## 🎨 Customization

Edit parameters in `main.py`:

```python
OUTPUTDIR = "output"          # Output folder
FPS = 24                      # Video frame rate
QUIZSECONDS = 10             # Quiz duration (seconds)
ANSWERSECONDS = 2            # Answer reveal duration
MAX_DISPLAY = 1024           # Max preview size (pixels)
MAX_VIDEO_WIDTH = 540        # Video width limit (prevents OOM)
```

## 🛠️ Tech Stack

- **Gradio 3.50.2** - Web UI framework
- **OpenCV** - Image processing and video generation
- **Pillow** - Image manipulation
- **MoviePy** - Video encoding with FFmpeg
- **NumPy** - Array operations

## 📦 Project Structure

```
spotdiff/
├── main.py              # Main Gradio app with video generation
├── main_opencv.py       # Alternative OpenCV-based implementation
├── app.py               # Additional app utilities
├── index.html           # Custom HTML (if needed)
├── requirements.txt     # Python dependencies
└── output/              # Generated videos and temp files
```

## 🚀 Deployment

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Choose **Gradio** as SDK
3. Upload your code
4. Set Python version to 3.8+
5. Add `ffmpeg` as a system dependency

### Deploy to Render

1. Connect your GitHub repository
2. Create a new Web Service
3. Build command: `pip install -r requirements.txt`
4. Start command: `python main.py`
5. Set environment variable: `PORT=7860`

### Deploy to Railway

```bash
railway up
```

Make sure to install FFmpeg in your deployment environment.

## 🎯 Use Cases

- **YouTube Shorts**: Create engaging "Find the Difference" challenges
- **TikTok Content**: Quick puzzle videos for viral content
- **Instagram Reels**: Interactive visual puzzles
- **Educational Content**: Visual attention and observation training
- **Social Media Engagement**: Boost likes and comments with interactive content

## 🔧 Alternative Implementation

The repository includes `main_opencv.py` for pure OpenCV-based video generation if you prefer not to use MoviePy.

## 📝 Tips for Best Results

1. **Image Quality**: Use high-resolution images (at least 1080p)
2. **Difference Visibility**: Make differences noticeable but not too obvious
3. **Circle Size**: Adjust circle radius to match difference size
4. **Contrast**: Ensure good contrast between base and variant images
5. **Vertical Format**: Use portrait-oriented images for better Shorts layout

## ⚠️ Troubleshooting

### Out of Memory Errors
- Reduce `MAX_VIDEO_WIDTH` in `main.py`
- Use smaller input images
- Close other applications

### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

### Video Generation Fails
- Ensure images are in RGB format
- Check that both images are valid
- Verify FFmpeg is installed correctly

## 📄 License

MIT License - feel free to use this project for any purpose.

## 👤 Author

**Phaapnag** - [GitHub](https://github.com/Phaapnag)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

Built with ❤️ for content creators