# import pytesseract
# from PIL import Image
# import cv2

# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# img = cv2.imread("text123.png")

# if img is None:
#     print("Error: Image not found or unable to read.")
#     exit()
    
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

# blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# cv2.imwrite("processed.jpg", thresh)

# processed_image = Image.fromarray(thresh)

# custom_config = r'--oem 3 --psm 6'
# text = pytesseract.image_to_string(processed_image, config=custom_config)

# print(text) 

# ===================================================================================== this is image extract text without streamlit ============================

# import streamlit as st
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np
# import tempfile
# import os
# import shutil
# import subprocess
# from langdetect import detect, DetectorFactory

# tesseract_path = shutil.which("tesseract")
# if tesseract_path:
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path
# else:
#     st.error("❌ Tesseract not found! Please install it and add to your system PATH.")
#     st.stop()

# st.title("🧠 OCR Text Extraction from Image")

# uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#     # Save the uploaded file to a temporary location
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_path = tmp_file.name

#     # Load the image using OpenCV
#     img = cv2.imread(tmp_path)

#     if img is None:
#         st.error("Error: Unable to read the uploaded image.")
#     else:
#         # Preprocess the image
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Resize for better OCR
#         resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

#         # Apply Adaptive Thresholding (correct method)
#         thresh = cv2.adaptiveThreshold(resized, 255,
#                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY,
#                                        31, 11)

#         # Convert processed image to RGB for displaying
#         processed_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

#         st.markdown("### 🔍 Processed Image")
#         st.image(processed_rgb, caption="Processed for OCR", use_container_width=True)

#         # OCR with Tesseract
#         custom_config = r'--psm 6 -l eng'
#         processed_image = Image.fromarray(thresh)
#         extracted_text = pytesseract.image_to_string(processed_image, config=custom_config) 
#         try:
#             lang_detected = detect(extracted_text)
#         except:
#             lang_detected = "unknown"

#         st.markdown(f"### 🌍 Detected Language: `{lang_detected}`")

#         st.markdown("### 📄 Extracted Text")
#         st.text_area("Text", extracted_text, height=800)

#         # Offer download option
#         st.download_button("Download Extracted Text", extracted_text, file_name="extracted_text.txt")

#         # Clean up temporary file
#         os.remove(tmp_path)

# else:
#     st.info("Upload an image to start OCR processing.")
# =========================================================================================== 

# import streamlit as st
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np
# import tempfile
# from langdetect import detect

# st.title("🌐 OCR with Auto Language Detection")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

#     # Save the uploaded file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_path = tmp_file.name

#     # Load and preprocess
#     img = cv2.imread(tmp_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
#     thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 31, 11)

#     st.markdown("### 🔍 Processed Image")
#     st.image(thresh, use_container_width=True)

#     # Run OCR using English
#     custom_config = r'--oem 3 --psm 6 -l  eng+hin+nep+guj+ben+mar+san'
#     text = pytesseract.image_to_string(thresh, config=custom_config)

#     # Detect language
#     try:
#         detected_lang = detect(text)
#     except:
#         detected_lang = "unknown"

#     st.markdown("### 📄 Extracted Text")
#     st.text_area("Text", text, height=800)

#     st.markdown(f"**🈯 Detected Language:** `{detected_lang}`")

#     st.download_button("📥 Download Extracted Text", text, file_name="extracted_text.txt")

# else:
#     st.info("📤 Upload an image to begin.")
# ==============================================================================================

# import streamlit as st
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np
# import tempfile
# import os
# import shutil
# from langdetect import detect, DetectorFactory

# DetectorFactory.seed = 0

# # Set Tesseract path
# # Set tesseract path and tessdata folder
# tesseract_path = shutil.which("tesseract")

# if tesseract_path:
#     pytesseract.pytesseract.tesseract_cmd = tesseract_path

#     # Get tesseract directory and point to tessdata folder
#     tesseract_dir = os.path.dirname(tesseract_path)
#     tessdata_dir = os.path.join(tesseract_dir, 'tessdata')
#     os.environ["TESSDATA_PREFIX"] = tessdata_dir
# else:
#     st.error("❌ Tesseract not found. Please install Tesseract OCR.")
#     st.stop()

# st.title("📄 OCR with Language Detection and Multilingual Support")

# uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])   

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True) 

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_path = tmp_file.name

#     # Load and preprocess image
#     img = cv2.imread(tmp_path)
#     if img is None:
#         st.error("❌ Could not load the image.")
#     else:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
#         thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                        cv2.THRESH_BINARY, 31, 11)

#         st.markdown("### 🧪 Processed Image")
#         st.image(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB), use_container_width=True)

#         # First pass OCR using English as fallback
#         fallback_text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 6', lang='eng')

#         # Detect language
#         try:
#             detected_lang_code = detect(fallback_text.strip())
#         except:
#             detected_lang_code = "unknown"

#         # Get available Tesseract languages
#         available_langs = pytesseract.get_languages(config='')

#         # Try to match detected language with Tesseract languages
#         lang_map = {
#             "gu": "guj",
#             "hi": "hin",
#             "en": "eng",
#             "bn": "ben",
#             "ta": "tam",
#             "mr": "mar"
#             # Add more as needed
#         }

#         tess_lang = lang_map.get(detected_lang_code, "eng")
#         if tess_lang not in available_langs:
#             tess_lang = "eng"  # fallback

#         # Final OCR using detected language
#         try:
#             extracted_text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 6', lang=tess_lang)
#         except pytesseract.TesseractError as e:
#             st.error(f"❌ OCR Error: {e}")
#             extracted_text = ""

#         st.markdown(f"### 🌍 Detected Language: `{detected_lang_code}` (Tesseract: `{tess_lang}`)")
#         st.markdown("### 📄 Extracted Text")
#         st.text_area("Text", extracted_text, height=600)
#         st.download_button("📥 Download Text", extracted_text, file_name="ocr_text.txt")

#     os.remove(tmp_path)
# else:
#     st.info("📤 Upload an image to extract text.")



# =======================================================================================
# import streamlit as st
# from PIL import Image
# import pytesseract
# import cv2
# import numpy as np

# st.title("🧾 Invoice OCR Extractor")

# uploaded_file = st.file_uploader("Upload an invoice image", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded Invoice", use_container_width=True)

#     image = Image.open(uploaded_file)
#     open_cv_image = np.array(image.convert('RGB'))
#     gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
#     thresh = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
#     )

#     # OCR using PSM 6 (assumes uniform text blocks)
#     custom_config = r'--psm 6 -l eng'
#     text = pytesseract.image_to_string(thresh, config=custom_config)

#     if text.strip():
#         st.subheader("📋 Extracted Text")
#         st.text_area("OCR Output", text, height=300)
#     else:
#         st.error("No text could be extracted. Try with another image.")


import os
os.environ['TESSDATA_PREFIX'] = r'C:\\Program Files\\Tesseract-OCR\\tessdata\\'

from PIL import Image
import pytesseract

img = Image.open('Tax-Invoice-Format-in-Excel.jpg')
text = pytesseract.image_to_string(img, lang='mar')
print(text)