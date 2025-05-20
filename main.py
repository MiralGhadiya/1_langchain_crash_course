# import cv2
# import pytesseract
# import os

# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# VIDEO_PATH = "news_small.mp4"
# OUTPUT_FILE = "output_text.txt"
# FRAME_INTERVAL = 100

# def preprocess_frame(frame):
#     """Enhance frame for better OCR accuracy."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  
#     blur = cv2.medianBlur(gray, 3)
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )
#     return thresh

# def extract_text_from_video(video_path, frame_interval, output_file):
#     if not os.path.exists(video_path):
#         print(f"Video file '{video_path}' not found!")
#         return

#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     current_frame = 0
#     frame_count = 0

#     with open(output_file, "w", encoding="utf-8") as f:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if current_frame % frame_interval == 0:
#                 print(f"Processing frame {current_frame}/{total_frames}...")
#                 processed_frame = preprocess_frame(frame)
#                 text = pytesseract.image_to_string(processed_frame)
#                 cleaned_text = text.strip()
#                 if cleaned_text:
#                     f.write(f"\n--- Frame {current_frame} ---\n{cleaned_text}\n")
#                     frame_count += 1
#                 # f.write(f"\n--- Frame {current_frame} ---\n{text}\n")
#                 # frame_count += 1  

#             current_frame += 1

#     cap.release()
#     print(f"\n✅ Done! Extracted text from {frame_count} frames and saved to '{output_file}'")

# if __name__ == "__main__":
#     extract_text_from_video(VIDEO_PATH, FRAME_INTERVAL, OUTPUT_FILE)
# ============================================================================= this is simple code ==============================

# import streamlit as st
# import cv2
# import pytesseract
# import os
# import tempfile
# import shutil
# import sys


# tesseract_path = shutil.which("tesseract")
# if tesseract_path:
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
# else:
#     sys.exit("Tesseract executable not found. Please install Tesseract or add it to your PATH.")

# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     blur = cv2.medianBlur(gray, 3)
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )
#     return thresh

# # OCR Extraction function
# def extract_text_from_video(video_path, frame_interval):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         st.error("Error opening video file.")
#         return "", 0

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     current_frame = 0
#     frame_count = 0
#     output_text = ""

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if current_frame % frame_interval == 0:
#             processed_frame = preprocess_frame(frame)
#             text = pytesseract.image_to_string(processed_frame)
#             cleaned_text = text.strip()
#             if cleaned_text:
#                 output_text += f"\n--- Frame {current_frame} ---\n{cleaned_text}\n"
#                 frame_count += 1

#                 # Show original frame as image in Streamlit
#                 st.markdown(f"### Frame {current_frame}")
#                 rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 st.image(rgb_frame, caption=f"Frame {current_frame}", use_container_width=True)

#         current_frame += 1

#         # Update progress
#         if current_frame % 100 == 0 or current_frame == total_frames:
#             progress = min(current_frame / total_frames, 1.0)
#             progress_bar.progress(progress)
#             status_text.text(f"Processing frame {current_frame} / {total_frames}")

#     cap.release()
#     progress_bar.empty()
#     status_text.empty()

#     return output_text, frame_count

# # Streamlit UI
# st.title("Video Text Extractor with OCR")

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
# frame_interval = st.number_input("Frame interval (process every N-th frame)", min_value=1, value=100)

# if uploaded_file:
#     st.video(uploaded_file)

#     if st.button("Start OCR Extraction"):
#         with st.spinner("Processing..."):
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(uploaded_file.read())
#             video_path = tfile.name

#             extracted_text, frames_used = extract_text_from_video(video_path, frame_interval)

#             if extracted_text:
#                 st.success(f"Done! Extracted text from {frames_used} frames.")
#                 st.text_area("Extracted Text", extracted_text, height=300)

#                 # Download button
#                 st.download_button("Download Text", data=extracted_text, file_name="output_text.txt")
#             else:
#                 st.warning("No text detected in the selected frames.")

# ====================================================== this is my streamlit without screenshot =========================================================


# import streamlit as st
# import cv2
# import pytesseract
# import os
# import tempfile
# import shutil
# import sys

# tesseract_path = shutil.which("tesseract")
# if tesseract_path:
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
# else:
#     st.error("Tesseract executable not found. Please install Tesseract or add it to your PATH.")
#     st.stop()

# def preprocess_frame(frame):
#     """Enhance frame for better OCR accuracy."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
#     blur = cv2.medianBlur(gray, 3)
#     thresh = cv2.adaptiveThreshold(
#         blur, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         11, 2
#     )
#     return thresh

# def extract_text_from_video(video_path, frame_interval, preview_frames=False, max_preview=5):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         st.error("Failed to open video file.")
#         return "", 0, []

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     output_text = ""
#     frames_used = 0
#     current_frame = 0
#     preview_images = []

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if current_frame % frame_interval == 0:
#             processed_frame = preprocess_frame(frame)
#             text = pytesseract.image_to_string(processed_frame).strip()

#             if text:
#                 output_text += f"\n--- Frame {current_frame} ---\n{text}\n"
#                 frames_used += 1

#                 if preview_frames and len(preview_images) < max_preview:
#                     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     preview_images.append((current_frame, rgb_frame))

#             status_text.text(f"Processing frame {current_frame}/{total_frames}...")

#         current_frame += 1

#         progress = min(current_frame / total_frames, 1.0)
#         progress_bar.progress(progress)

#     cap.release()
#     progress_bar.empty()
#     status_text.empty()

#     return output_text, frames_used, preview_images

# st.title("📹 Video Text Extractor with OCR")

# with st.sidebar:
#     st.header("Settings")
#     frame_interval = st.number_input("Frame interval (process every N-th frame)", min_value=1, max_value=500, value=100, step=10)
#     preview_toggle = st.checkbox("Show preview of sampled frames", value=True)
#     max_preview = st.slider("Max preview frames to show", 1, 10, 5)

# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

# if uploaded_file:
#     st.video(uploaded_file)

#     if st.button("Start OCR Extraction"):
#         with st.spinner("Processing video and extracting text..."):

#             with tempfile.NamedTemporaryFile(delete=False) as tfile:
#                 tfile.write(uploaded_file.read())
#                 video_path = tfile.name

#             extracted_text, frames_used, previews = extract_text_from_video(
#                 video_path, frame_interval, preview_frames=preview_toggle, max_preview=max_preview
#             )

#             os.remove(video_path)

#             if frames_used > 0:
#                 st.success(f" Extraction done! Text extracted from {frames_used} frames.")
#                 st.text_area("Extracted Text", extracted_text, height=300)

#                 if preview_toggle and previews:
#                     st.markdown("### Sampled Frame Previews:")
#                     for frame_no, img in previews:
#                         st.image(img, caption=f"Frame {frame_no}", use_container_width=True)
                
#                 st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt")
#             else:
#                 st.warning("No text detected in the selected frames.")

# else:
#     st.info("Please upload a video file to begin OCR extraction.")

# ========================================================================= this is my code, I am trying to make a simple video text extractor with OCR using streamlit and pytesseract ===============


import streamlit as st
import cv2
import pytesseract
import tempfile
import shutil
import os


tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("❌ Tesseract not found! Please install it and add to your system PATH.")
    st.stop()


def preprocess_frame(frame):
    """
    Convert a frame to grayscale, resize, blur and apply adaptive thresholding.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def extract_text_from_video(video_path, frame_interval, preview_frames=False, max_preview=5, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("🚫 Unable to open video.")
        return "", 0, []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_text = ""
    previews = []
    frames_used = 0
    current_frame = 0

    progress = st.progress(0)
    status = st.empty()

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % frame_interval == 0:
            processed = preprocess_frame(frame)
            text = pytesseract.image_to_string(processed).strip()

            if text:
                output_text += f"\n--- Frame {current_frame} ---\n{text}\n"
                frames_used += 1

                if preview_frames and len(previews) < max_preview:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    previews.append((current_frame, rgb))

            if max_frames and frames_used >= max_frames:
                break

        progress.progress(min(current_frame / total_frames, 1.0))
        status.text(f"🔄 Processing frame {current_frame} of {total_frames}")
        current_frame += 1

    cap.release()
    progress.empty()
    status.empty()
    return output_text, frames_used, previews


st.set_page_config(page_title="Video OCR", layout="centered")
st.title("🎥 Video Text Extractor using OCR")

with st.sidebar:
    st.header("⚙️ Settings")
    frame_interval = st.slider("Process every N-th frame", 1, 300, 50)
    preview_frames = st.checkbox("Show preview frames", value=True)
    max_preview = st.slider("Max preview frames", 1, 10, 5)
    max_process = st.number_input("Max frames to process (0 = no limit)", 0, 5000, 0)

uploaded_file = st.file_uploader("📁 Upload a video file", type=["mp4", "mov", "avi"])
if uploaded_file:
    st.video(uploaded_file)

    if st.button("🚀 Start OCR Extraction"):
        with st.spinner("Analyzing video... Please wait."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_vid:
                temp_vid.write(uploaded_file.read())
                temp_path = temp_vid.name

            extracted_text, frames_used, previews = extract_text_from_video(
                temp_path,
                frame_interval=frame_interval,
                preview_frames=preview_frames,
                max_preview=max_preview,
                max_frames=max_process if max_process > 0 else None
            )

            os.remove(temp_path)

        if frames_used > 0:
            st.success(f"✅ Done! Extracted text from {frames_used} frames.")
            st.text_area("📄 Extracted Text", extracted_text, height=300)

            if previews:
                st.markdown("### 🖼️ Preview Frames")
                for frame_no, img in previews:
                    st.image(img, caption=f"Frame {frame_no}", use_container_width=True)

            st.download_button("💾 Download Text", extracted_text, file_name="extracted_text.txt")
        else:
            st.warning("⚠️ No text found in the selected frames.")
else:
    st.info("⬆️ Upload a video file to begin.")
