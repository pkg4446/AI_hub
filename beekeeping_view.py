from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch

# 1. 모델 로드
model = YOLO('yolo11l.pt')

# 2. 이미지에서 객체 탐지
def detect_objects_image(image_path):
    """이미지에서 객체 탐지"""
    results = model(image_path)
    
    # 결과 시각화
    annotated_img = results[0].plot()
    
    # 결과 저장
    cv2.imwrite('result.jpg', annotated_img)
    
    # 탐지된 객체 정보 출력
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = box.xyxy[0]
            # 신뢰도
            confidence = box.conf[0]
            # 클래스 ID
            cls = box.cls[0]
            # 클래스 이름
            class_name = model.names[int(cls)]
            
            print(f"객체: {class_name}, 신뢰도: {confidence:.2f}, 좌표: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    
    return results

# 3. 웹캠 실시간 탐지
def detect_webcam():
    """웹캠으로 실시간 객체 탐지"""
    cap = cv2.VideoCapture(0)    
    while True:
        ret, frame = cap.read()
        if not ret:
            break        
        # 객체 탐지
        results = model(frame)        
        # 결과 시각화
        annotated_frame = results[0].plot()        
        # 화면에 표시
        cv2.imshow('YOLO11 Detection', annotated_frame)        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
    cap.release()
    cv2.destroyAllWindows()

# 4. 커스텀 신뢰도 및 필터링
def detect_with_custom_settings(image_path, conf_threshold=0.5, classes=None):
    """커스텀 설정으로 객체 탐지"""
    results = model(image_path, conf=conf_threshold, classes=classes)
    
    # 결과 시각화
    annotated_img = results[0].plot()
    cv2.imwrite('custom_result.jpg', annotated_img)
    
    return results

# 5. 탐지 결과를 JSON으로 저장
def save_results_to_json(image_path, output_json='detection_results.json'):
    """탐지 결과를 JSON 파일로 저장"""
    import json
    
    results = model(image_path)
    
    detection_data = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]
            
            detection_data.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2)
    
    print(f"탐지 결과가 {output_json}에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 이미지 탐지
    detect_objects_image('beekeeping_image.jpg')
    
    # 웹캠 실시간 탐지
    # detect_webcam()
        
    # 특정 클래스만 탐지 (예: egg(0),lava(1),pupa(2),bee(3),queen(4))
    # detect_with_custom_settings('beekeeping_image.jpg', conf_threshold=0.6, classes=[0,1,2])
    
    # 결과를 JSON으로 저장
    save_results_to_json('beekeeping_image.jpg')
    
    print("YOLO11 탐지 완료!")