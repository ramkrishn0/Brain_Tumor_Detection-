import streamlit as st
from ultralytics import YOLO
import base64
import io
from PIL import Image, ImageDraw, ImageFont

# Load the model once
@st.cache_resource
def load_model():
    return YOLO("best.pt", task='yolov5s') # Change the location of the model.

model_data = load_model()

# Streamlit UI for file upload
st.title("YOLO Object Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    image = Image.open(uploaded_file)
    image_path = rf"images/{uploaded_file.name}" # Change the location of image that you saved
    image.save(image_path)

    # Run object detection
    result = model_data(image_path)

    if result:
        detections = []
        image_with_boxes = Image.open(image_path)
        draw = ImageDraw.Draw(image_with_boxes)
        font = ImageFont.truetype("arial.ttf", 20)

        for res in result:
            if res.boxes is not None:
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
                    text = f"{res.names[int(box.cls.item())]} {float(box.conf)*100:.2f}%"
                    draw.text((x1, y1), text, fill="yellow", font=font)
                    detections.append(text)

        if detections:
            img_byte_arr = io.BytesIO()
            image_with_boxes.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            st.image(image_with_boxes, caption="Object Detection Result")
            st.write("Detections:")
            for detection in detections:
                st.write(f"- {detection}")
        else:
            st.write("No Objects Detected")

    else:
        st.write("No Objects Detected")
