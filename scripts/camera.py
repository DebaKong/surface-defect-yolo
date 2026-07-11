import cv2
from ultralytics import YOLO
import time
import os
import csv
import datetime

# 加载YOLO模型
model = YOLO("D:\\Visual Studio Code 2025\\VS code project\\best.pt")  # 替换为你的模型路径

# 打开摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print('摄像头启动成功')

# 初始化变量
prev_time = 0

last_alarm_time = 0

alarm_interval = 5

# 保存路径
alarm_dir = "D:\\Visual Studio Code 2025\\VS code project\\alarm_images"
if not os.path.exists(alarm_dir):
    os.makedirs(alarm_dir)

log_file = "D:\\Visual Studio Code 2025\\VS code project\\logs.csv"
if not os.path.exists(log_file):
    with open(log_file,"w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time","class","confidence"]
        )

# 主循环
while True:
    # 读取摄像头帧
    ret, frame = cap.read()
    
    roi=frame[100:500,200:800]  # 裁剪检测区域（根据需要调整坐标）
    
    if not ret:
        print("无法读取摄像头帧")
        break

    # 使用YOLO模型进行目标检测
    results = model(roi, imgsz=640)  # 可以根据需要调整图像大小

    # 绘制检测结果
    annotated_frame = results[0].plot()  # 获取带有标注的帧

    # 显示FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # 检查是否检测到缺陷
    if  len(results[0].boxes) > 0:  # 如果检测到目标
        
        # 检查是否超过报警间隔
        if current_time - last_alarm_time > alarm_interval:
        
            cv2.putText(annotated_frame, "DEFECT DETECTED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 3)

            # 保存检测到缺陷的图像和日志
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                if confidence > 0.5:  # 只保存置信度大于0.5的检测结果
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(alarm_dir, f"defect_{timestamp}.jpg")
                    cv2.imwrite(image_path, annotated_frame)  # 保存带有标注的图像

                    # 写入日志
                    with open(log_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, model.names[class_id], round(confidence, 3)])
    
            last_alarm_time = current_time  # 更新最后报警时间
    
    # 显示结果
    cv2.imshow("YOLO Detection", annotated_frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
cap.release()
cv2.destroyAllWindows()