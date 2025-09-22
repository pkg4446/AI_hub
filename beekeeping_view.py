from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import os
from pathlib import Path

# 1. 모델 로드
model = YOLO('best.pt')

# 2. 객체 탐지만 수행하는 함수
def detect_objects_only(image_path, conf_threshold=0.5):
    """이미지에서 객체 탐지만 수행하고 결과 반환"""
    results = model(image_path, conf=conf_threshold)    
    return results

# 2-1. 탐지된 객체 정보를 출력하는 함수
def detect_infomation(results):
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0]
                # 신뢰도
                confidence = box.conf[0]
                # 클래스 ID
                cls = box.cls[0]
                # 클래스 이름
                class_name = model.names[int(cls)]
                
                detection_info = {
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'class_id': int(cls)
                }
                detections.append(detection_info)
                
                print(f"객체: {class_name}, 신뢰도: {confidence:.2f}, 좌표: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    
    return detections

# 3. 결과 이미지 저장하는 함수
def save_detection_image(results, image_path, output_dir, 
                        line_width=1, font_size=1, 
                        conf=True, labels=True, boxes=True, probs=True):
    """탐지 결과를 이미지로 저장"""
    # 결과 시각화
    annotated_img = results[0].plot(
        line_width=line_width,
        font_size=font_size,
        conf=conf,
        labels=labels,
        boxes=boxes,
        probs=probs
    )
    
    # 원본 파일명에서 확장자 제거하고 결과 파일명 생성
    image_name = Path(image_path).stem
    result_path = os.path.join(output_dir, f"{image_name}_result.jpg")
    
    # 결과 저장
    cv2.imwrite(result_path, annotated_img)
    print(f"결과 이미지 저장: {result_path}")
    
    return result_path

# 4. 통합된 객체 탐지 함수 (선택적 이미지 저장)
def detect_objects_image(image_path, output_dir=None, save_image=True, 
                        conf_threshold=0.5, **plot_kwargs):
    """이미지에서 객체 탐지하고 선택적으로 결과를 저장"""
    
    # 객체 탐지 수행
    results = detect_objects_only(image_path, conf_threshold)
    
    # 이미지 저장이 요청되고 출력 디렉토리가 지정된 경우에만 저장
    if save_image and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_path = save_detection_image(results, image_path, output_dir, **plot_kwargs)
        return results, result_path
    
    return results, None

# 5. 웹캠 실시간 탐지
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

# 6. 커스텀 신뢰도 및 필터링
def detect_with_custom_settings(image_path, conf_threshold=0.5, classes=None, save_image=True):
    """커스텀 설정으로 객체 탐지"""
    results = model(image_path, conf=conf_threshold, classes=classes)
    
    if save_image:
        # 결과 시각화
        annotated_img = results[0].plot()
        cv2.imwrite('custom_result.jpg', annotated_img)
    
    return results

# 7. 탐지 결과를 YOLO 어노테이션으로 저장
def save_results_to_yolo_annotation(results, image_path, output_dir):
    """탐지 결과를 YOLO 어노테이션 형식(.txt)으로 저장"""
    
    # 이미지 크기 가져오기
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    
    annotation_lines = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                # 바운딩 박스 좌표 (픽셀 단위)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                cls = int(box.cls[0].item())
                
                # YOLO 형식으로 변환 (정규화된 중심점과 너비, 높이)
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # YOLO 어노테이션 라인 생성
                annotation_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 원본 파일명에서 확장자 제거하고 txt 파일명 생성
    image_name = Path(image_path).stem
    txt_path = os.path.join(output_dir, f"{image_name}.txt")
    
    # YOLO 어노테이션 저장
    with open(txt_path, 'w', encoding='utf-8') as f:
        for line in annotation_lines:
            f.write(line + '\n')
    
    print(f"YOLO 어노테이션 저장: {txt_path}")
    return annotation_lines

# 8. 클래스 이름을 classes.txt 파일로 저장
def save_class_names(output_dir):
    """클래스 이름을 classes.txt 파일로 저장"""
    classes_path = os.path.join(output_dir, 'classes.txt')
    
    with open(classes_path, 'w', encoding='utf-8') as f:
        for class_id, class_name in model.names.items():
            f.write(f"{class_name}\n")
    
    print(f"클래스 이름 저장: {classes_path}")

# 9. 폴더 내 모든 이미지 처리하는 함수 (개선됨)
def process_all_images(input_dir='./image', 
                      result_dir='./result', 
                      annotation_dir='./label',
                      save_images=True,
                      save_annotations=True,
                      conf_threshold=0.5):
    """폴더 내 모든 이미지를 처리하여 선택적으로 결과를 저장"""
    
    # 출력 폴더들이 존재하지 않으면 생성
    if save_images:
        os.makedirs(result_dir, exist_ok=True)
    if save_annotations:
        os.makedirs(annotation_dir, exist_ok=True)
        # 클래스 이름 저장
        save_class_names(annotation_dir)
    
    # 지원하는 이미지 확장자
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # 입력 폴더에서 이미지 파일 찾기
    image_files = []
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(input_dir, file))
    else:
        print(f"입력 폴더 '{input_dir}'가 존재하지 않습니다.")
        return
    
    if not image_files:
        print(f"'{input_dir}' 폴더에 이미지 파일이 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 처리합니다.")
    print(f"이미지 저장: {'예' if save_images else '아니오'}")
    print(f"어노테이션 저장: {'예' if save_annotations else '아니오'}")
    
    # 각 이미지 파일 처리
    all_detections = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 처리 중: {os.path.basename(image_path)}")
        
        try:
            # 객체 탐지 수행
            results, result_path = detect_objects_image(
                image_path, 
                result_dir if save_images else None, 
                save_images, 
                conf_threshold
            )
            
            # YOLO 어노테이션 저장 (요청된 경우)
            if save_annotations:
                save_results_to_yolo_annotation(results, image_path, annotation_dir)
            
        except Exception as e:
            print(f"오류 발생 ({os.path.basename(image_path)}): {str(e)}")
            continue
    
    print(f"\n모든 처리가 완료되었습니다!")
    print(f"총 탐지된 객체 수: {len(all_detections)}")
    
    if save_images:
        print(f"결과 이미지: {result_dir}")
    if save_annotations:
        print(f"YOLO 어노테이션: {annotation_dir}")
    
    return all_detections

# 사용 예시
if __name__ == "__main__":
    # 예시 1: 전체 이미지 처리
    print("=== 전체 이미지 처리 ===")
    process_all_images(save_images=False, save_annotations=True)

    # 예시 2: 개별 이미지 처리 - 탐지만
    # results, _ = detect_objects_image('./image/test.jpg', save_image=False)
    
    # 예시 3: 개별 이미지 처리 - 탐지 + 이미지 저장
    # results, result_path = detect_objects_image('./image/test.jpg', './result', save_image=True)
    
    # 웹캠 실시간 탐지
    # detect_webcam()
        
    # 특정 클래스만 탐지 (예: egg(0),lava(1),pupa(2),bee(3),queen(4))
    # detect_with_custom_settings('./image/beekeeping_image.jpg', conf_threshold=0.6, classes=[0,1,2], save_image=False)
    
    print("YOLO11 배치 처리 완료!")