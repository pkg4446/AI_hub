# YOLO v11 커스텀 데이터셋 학습

# 1. 패키지 설치 (터미널에서 실행)
# pip install ultralytics
# pip install torch matplotlib pandas opencv-python

# 라이브러리 import
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from ultralytics import YOLO
import glob
import time

# GPU 확인
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
else:
    print("CPU를 사용합니다.")

# 2. 데이터셋 경로 설정 (실제 경로로 변경하세요)
DATASET_ROOT = "./dataset"  # 데이터셋 루트 폴더
PROJECT_NAME = "beekeeping"  # 프로젝트 이름
EXPERIMENT_NAME = "dreambee"  # 실험 이름

# 3. 데이터셋 구조 예시
"""
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/  # 선택적 테스트 세트
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    ├── val/
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── ...
    └── test/  # 선택적 테스트 라벨
        ├── img1.txt
        ├── img2.txt
        └── ...
"""

# 4. 데이터셋 YAML 파일 생성
def create_dataset_yaml(dataset_root, classes, yaml_path="dataset.yaml"):
    """데이터셋 YAML 파일 생성"""
    
    dataset_yaml = f"""# Train/val/test sets
path: {os.path.abspath(dataset_root)}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
"""
    
    # 클래스 추가
    for i, class_name in enumerate(classes):
        dataset_yaml += f"  {i}: {class_name}\n"
    
    dataset_yaml += f"\nnc: {len(classes)}  # number of classes"
    
    # YAML 파일 저장
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(dataset_yaml)
    
    print(f"Dataset YAML 파일이 생성되었습니다: {yaml_path}")
    return yaml_path

# 클래스 이름 설정
CLASS_NAMES = ['egg', 'lava', 'pupa', 'bee', 'queen', 'Chalkbrood']  # 클래스명
yaml_file = create_dataset_yaml(DATASET_ROOT, CLASS_NAMES)

# 5. 데이터셋 통계 확인 함수
def check_dataset_stats(dataset_root):
    """데이터셋 통계 확인"""
    train_images = glob.glob(os.path.join(dataset_root, 'images/train/*'))
    val_images = glob.glob(os.path.join(dataset_root, 'images/val/*'))
    train_labels = glob.glob(os.path.join(dataset_root, 'labels/train/*.txt'))
    val_labels = glob.glob(os.path.join(dataset_root, 'labels/val/*.txt'))
    
    print(f"=== 데이터셋 통계 ===")
    print(f"Train Images: {len(train_images)}")
    print(f"Train Labels: {len(train_labels)}")
    print(f"Val Images: {len(val_images)}")
    print(f"Val Labels: {len(val_labels)}")
    
    # 클래스별 객체 수 계산
    class_counts = {}
    total_objects = 0
    
    for label_file in train_labels + val_labels:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():  # 빈 줄이 아닌 경우
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
        except Exception as e:
            print(f"라벨 파일 읽기 오류 {label_file}: {e}")
    
    print(f"\n=== 클래스별 객체 수 ===")
    for class_id, count in sorted(class_counts.items()):
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Unknown_{class_id}"
        print(f"Class {class_id} ({class_name}): {count}개")
    print(f"총 객체 수: {total_objects}개")
    
    return len(train_images), len(val_images), total_objects

# 데이터셋 통계 확인
if os.path.exists(DATASET_ROOT):
    train_count, val_count, obj_count = check_dataset_stats(DATASET_ROOT)
else:
    print(f"데이터셋 폴더를 찾을 수 없습니다: {DATASET_ROOT}")
    print("DATASET_ROOT 변수를 실제 데이터셋 경로로 변경해주세요.")

# 6. 데이터셋 샘플 시각화 함수
def visualize_dataset_sample(dataset_root, num_samples=4):
    """데이터셋 샘플 시각화"""
    train_images = glob.glob(os.path.join(dataset_root, 'images/train/*'))[:num_samples]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, img_path in enumerate(train_images):
        if idx >= num_samples:
            break
            
        # 이미지 로드
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 라벨 파일 경로
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(dataset_root, 'labels/train', f'{base_name}.txt')
        
        axes[idx].imshow(img)
        axes[idx].set_title(f'Sample {idx+1}: {os.path.basename(img_path)}')
        axes[idx].axis('off')
        
        # 라벨이 있으면 바운딩 박스 그리기
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    labels = f.readlines()
                
                for label in labels:
                    if label.strip():
                        parts = label.strip().split()
                        if len(parts) >= 5:
                            class_id, x_center, y_center, width, height = map(float, parts[:5])
                            
                            # YOLO 형식을 pixel 좌표로 변환
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            
                            # 바운딩 박스 그리기
                            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               fill=False, color='red', linewidth=2)
                            axes[idx].add_patch(rect)
                            
                            # 클래스 라벨 표시
                            class_name = CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f'Class_{int(class_id)}'
                            axes[idx].text(x1, y1-10, class_name, 
                                         color='red', fontsize=10, weight='bold',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
            except Exception as e:
                print(f"라벨 시각화 오류 {label_path}: {e}")
    
    plt.tight_layout()
    plt.show()

# 샘플 시각화 (데이터셋이 있을 때만)
if os.path.exists(DATASET_ROOT):
    visualize_dataset_sample(DATASET_ROOT)

# 7. YOLO v11 모델 학습
def train_yolo_model(yaml_path, project_name, experiment_name):
    """YOLO v11 모델 학습"""
    
    # 사전 훈련된 모델 로드
    # model = YOLO('yolo11n.pt')  # nano 버전 (빠름, 정확도 낮음)
    # model = YOLO('yolo11s.pt')  # small 버전 (균형)
    # model = YOLO('yolo11m.pt')  # medium 버전 (느림, 정확도 높음)
    model = YOLO('yolo11l.pt')  # large 버전 (매우 느림, 매우 높은 정확도)
    
    print("모델 학습을 시작합니다...")
    
    # 학습 실행
    results = model.train(
        data=yaml_path,               # 데이터셋 yaml 파일 경로
        epochs=100,                   # 학습 에포크 수
        imgsz=640,                    # 이미지 크기 (로컬에서는 640 권장)
        batch=16,                     # 배치 크기 (GPU 메모리에 따라 조정)
        device='0' if torch.cuda.is_available() else 'cpu',  # GPU/CPU 자동 선택
        workers=4,                    # 데이터 로더 워커 수
        project=project_name,         # 프로젝트 폴더명
        name=experiment_name,         # 실험명
        save=True,                    # 모델 저장
        save_period=5,                # N 에포크마다 체크포인트 저장
        patience=20,                  # Early stopping patience
        resume=False,                 # 중단된 학습 재개시 True
        amp=True,                     # Automatic Mixed Precision
        # 하이퍼파라미터
        lr0=0.01,                     # 초기 학습률
        weight_decay=0.0005,          # 가중치 감쇠
        warmup_epochs=3,              # Warmup 에포크
        # 데이터 증강
        hsv_h=0.015,                  # 색조 증강
        hsv_s=0.7,                    # 채도 증강
        hsv_v=0.4,                    # 명도 증강
        degrees=0.0,                  # 회전 증강
        translate=0.1,                # 이동 증강
        scale=0.5,                    # 스케일 증강
        shear=0.0,                    # 전단 증강
        flipud=0.0,                   # 수직 플립
        fliplr=0.5,                   # 수평 플립
        mosaic=1.0,                   # 모자이크 증강
        mixup=0.0,                    # 믹스업 증강
    )
    
    return results

# 8. 학습 결과 시각화 함수
def plot_training_results(results_path):
    """학습 결과 그래프 그리기"""
    csv_path = os.path.join(results_path, 'results.csv')
    
    if not os.path.exists(csv_path):
        print(f"결과 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    try:
        # results.csv 파일 읽기
        results_df = pd.read_csv(csv_path)
        results_df.columns = results_df.columns.str.strip()  # 공백 제거
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO v11 Training Progress', fontsize=16)
        
        epochs = results_df['epoch']
        
        # 1. Training Loss 그래프
        axes[0,0].plot(epochs, results_df['train/box_loss'], label='Box Loss', color='blue')
        axes[0,0].plot(epochs, results_df['train/cls_loss'], label='Class Loss', color='red')
        axes[0,0].plot(epochs, results_df['train/dfl_loss'], label='DFL Loss', color='green')
        axes[0,0].set_title('Training Losses')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Validation Loss 그래프
        axes[0,1].plot(epochs, results_df['val/box_loss'], label='Box Loss', color='blue')
        axes[0,1].plot(epochs, results_df['val/cls_loss'], label='Class Loss', color='red')
        axes[0,1].plot(epochs, results_df['val/dfl_loss'], label='DFL Loss', color='green')
        axes[0,1].set_title('Validation Losses')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. mAP 그래프
        axes[1,0].plot(epochs, results_df['metrics/mAP50(B)'], label='mAP@0.5', color='purple')
        axes[1,0].plot(epochs, results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        axes[1,0].set_title('Mean Average Precision')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('mAP')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # 4. Precision & Recall 그래프
        axes[1,1].plot(epochs, results_df['metrics/precision(B)'], label='Precision', color='green')
        axes[1,1].plot(epochs, results_df['metrics/recall(B)'], label='Recall', color='red')
        axes[1,1].set_title('Precision & Recall')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Score')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 최신 성능 출력
        latest = results_df.iloc[-1]
        print(f"\n=== 최종 학습 결과 (Epoch {int(latest['epoch'])}) ===")
        print(f"Train Box Loss: {latest['train/box_loss']:.4f}")
        print(f"Train Class Loss: {latest['train/cls_loss']:.4f}")
        print(f"Val Box Loss: {latest['val/box_loss']:.4f}")
        print(f"Val Class Loss: {latest['val/cls_loss']:.4f}")
        print(f"mAP@0.5: {latest['metrics/mAP50(B)']:.4f}")
        print(f"mAP@0.5:0.95: {latest['metrics/mAP50-95(B)']:.4f}")
        print(f"Precision: {latest['metrics/precision(B)']:.4f}")
        print(f"Recall: {latest['metrics/recall(B)']:.4f}")
        
    except Exception as e:
        print(f"그래프 생성 중 오류: {e}")

# 9. 모델 평가 및 추론 함수
def evaluate_and_test_model(results_path, yaml_path, test_image_path=None):
    """모델 평가 및 테스트"""
    best_model_path = os.path.join(results_path, 'weights', 'best.pt')
    
    if not os.path.exists(best_model_path):
        print(f"학습된 모델을 찾을 수 없습니다: {best_model_path}")
        return
    
    # 최고 성능 모델 로드
    model = YOLO(best_model_path)
    
    # 검증 데이터에 대한 평가
    print("모델 평가 중...")
    val_results = model.val(data=yaml_path)
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # 테스트 이미지 추론
    if test_image_path and os.path.exists(test_image_path):
        print(f"테스트 이미지 추론: {test_image_path}")
        results = model(test_image_path, save=True)
        
        # 결과 표시
        for i, result in enumerate(results):
            # 결과 이미지 저장
            result.save(filename=f'inference_result_{i}.jpg')
            
            # 감지된 객체 정보 출력
            if result.boxes is not None:
                boxes = result.boxes
                print(f"감지된 객체 수: {len(boxes)}")
                for j, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'Class_{class_id}'
                    print(f"  객체 {j+1}: {class_name} (신뢰도: {confidence:.3f})")
    
    # 모델 내보내기
    print("모델을 ONNX 형식으로 내보내는 중...")
    model.export(format='onnx')
    print("ONNX 모델 내보내기 완료")
    
    return model

# 10. 메인 실행 함수
def main():
    """메인 실행 함수"""
    print("=== YOLO v11 커스텀 데이터셋 학습 시작 ===\n")
    
    # 데이터셋 존재 확인
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 데이터셋 폴더를 찾을 수 없습니다: {DATASET_ROOT}")
        print("DATASET_ROOT 변수를 실제 데이터셋 경로로 변경해주세요.")
        return
    
    # 학습 시작
    print("🚀 모델 학습 시작...")
    results = train_yolo_model(yaml_file, PROJECT_NAME, EXPERIMENT_NAME)
    
    # 결과 경로
    results_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME)
    
    # 학습 결과 시각화
    print("\n📊 학습 결과 시각화...")
    plot_training_results(results_path)
    
    # 테스트 경로 목록 준비
    test_paths = []
    if os.path.exists(os.path.join(DATASET_ROOT, 'images/test')):
        test_paths.append(os.path.join(DATASET_ROOT, 'images/test'))
    # 모델 평가
    print("\n🔍 모델 평가 및 테스트...")
    model = evaluate_and_test_model(results_path, yaml_file, test_paths)
    
    print(f"\n✅ 학습 완료!")
    print(f"📁 결과 폴더: {results_path}")
    print(f"🏆 최고 성능 모델: {os.path.join(results_path, 'weights', 'best.pt')}")
    print(f"💾 마지막 모델: {os.path.join(results_path, 'weights', 'last.pt')}")

# 실행
if __name__ == "__main__":
    # 개별 함수 실행 예시:
    
    # 1. 데이터셋만 확인하고 싶다면:
    # check_dataset_stats(DATASET_ROOT)
    # visualize_dataset_sample(DATASET_ROOT)
    
    # 2. 학습만 실행하고 싶다면:
    # train_yolo_model(yaml_file, PROJECT_NAME, EXPERIMENT_NAME)
    
    # 3. 기존 학습 결과만 시각화하고 싶다면:
    # plot_training_results('yolo_training/exp1')
    
    # 4. 전체 파이프라인 실행:
    main()