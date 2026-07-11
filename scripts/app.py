import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from datatime import datetime

# 加载模型
model = YOLO("D:\\Visual Studio Code 2025\\VS code project\\best.pt")  # 替换为你的模型路径

# 设置页面标题
st.title("YOLOv8 工业表面缺陷检测系统")

st.write("基于 YOLOv8 的工业钢材表面缺陷检测")

# 上传图片
uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取上传的图片
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # 显示原始图片
    st.image(image, caption="原始图片", use_column_width=True)

    # 将图片转换为 BGR 格式（OpenCV 使用）
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # 使用 YOLO 模型进行检测
    confidence = st.slider("置信度阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    results = model.predict(source=image_bgr, conf=confidence)

    # 获取检测结果
    result_image = results[0].plot()  # 绘制检测结果

    # 将结果图像从 BGR 转换回 RGB 格式
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # 显示检测结果
    st.image(result_image_rgb, caption="检测结果", use_column_width=True)

    # 显示检测结果的详细信息
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        st.write("检测到的缺陷数量:", len(boxes))
        for i, box in enumerate(boxes):
            st.write(f"缺陷 {i + 1}:")
            st.write(f"类别: {box.cls}")
            st.write(f"置信度: {box.conf:.2f}")
            st.write(f"边界框坐标: {box.xyxy}")

    else:
        st.write("未检测到缺陷。")
    
    # 结果保存功能
    save_image = Image.fromarray(result_image_rgb)
    time_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"D:\\Visual Studio Code 2025\\VS code project\\detect_result_{time_name}.jpg"  # 替换为你想保存的路径
    save_image.save(save_path)

    with open(save_path, "rb") as file:
        btn = st.download_button(
            label="下载检测结果",
            data=file,
            file_name="detect_result.jpg",
            mime="image/jpeg"
        )