# YOLO v11 학습 재개 및 추가 학습 (완전판)

import torch
import os
from ultralytics import YOLO
import glob
import yaml

# 기본 설정
DATASET_ROOT = "./dataset"
PROJECT_NAME = "beekeeping"
EXPERIMENT_NAME = "dreambee"
CLASS_NAMES = ['egg', 'lava', 'pupa', 'bee', 'queen', 'Chalkbrood']

def create_or_check_dataset_yaml(dataset_root, classes, yaml_path="dataset.yaml"):
    """데이터셋 YAML 파일 생성 또는 확인"""
    
    # YAML 파일이 이미 존재하는 경우 검증
    if os.path.exists(yaml_path):
        print(f"✅ 기존 dataset.yaml 파일 발견: {yaml_path}")
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                existing_config = yaml.safe_load(f)
            
            # 기존 설정 출력
            print(f"   - 데이터셋 경로: {existing_config.get('path')}")
            print(f"   - 클래스 수: {existing_config.get('nc')}")
            print(f"   - 클래스 이름: {list(existing_config.get('names', {}).values())}")
            
            # 클래스 수 일치 확인
            if existing_config.get('nc') != len(classes):
                print(f"⚠️  클래스 수 불일치: 기존 {existing_config.get('nc')} vs 현재 {len(classes)}")
                
            return yaml_path
            
        except Exception as e:
            print(f"❌ YAML 파일 읽기 오류: {e}")
            print("새로운 YAML 파일을 생성합니다...")
    
    # 새로운 YAML 파일 생성
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
    
    print(f"✅ Dataset YAML 파일 생성 완료: {yaml_path}")
    return yaml_path

def validate_dataset_structure(dataset_root):
    """데이터셋 구조 검증"""
    print(f"=== 데이터셋 구조 검증: {dataset_root} ===")
    
    if not os.path.exists(dataset_root):
        print(f"❌ 데이터셋 루트 폴더 없음: {dataset_root}")
        return False
    
    # 필수 폴더 확인
    required_folders = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    missing_folders = []
    for folder in required_folders:
        full_path = os.path.join(dataset_root, folder)
        if not os.path.exists(full_path):
            missing_folders.append(folder)
        else:
            # 파일 개수 확인
            file_count = len(glob.glob(os.path.join(full_path, '*')))
            print(f"✅ {folder}: {file_count}개 파일")
    
    if missing_folders:
        print(f"❌ 누락된 폴더: {missing_folders}")
        return False
    
    # 이미지와 라벨 매칭 확인
    train_images = set(os.path.splitext(os.path.basename(f))[0] 
                      for f in glob.glob(os.path.join(dataset_root, 'images/train/*')))
    train_labels = set(os.path.splitext(os.path.basename(f))[0] 
                      for f in glob.glob(os.path.join(dataset_root, 'labels/train/*.txt')))
    
    val_images = set(os.path.splitext(os.path.basename(f))[0] 
                    for f in glob.glob(os.path.join(dataset_root, 'images/val/*')))
    val_labels = set(os.path.splitext(os.path.basename(f))[0] 
                    for f in glob.glob(os.path.join(dataset_root, 'labels/val/*.txt')))
    
    # 매칭 상태 확인
    train_missing_labels = train_images - train_labels
    train_missing_images = train_labels - train_images
    val_missing_labels = val_images - val_labels
    val_missing_images = val_labels - val_images
    
    if train_missing_labels:
        print(f"⚠️  훈련 세트 라벨 누락: {len(train_missing_labels)}개")
    if train_missing_images:
        print(f"⚠️  훈련 세트 이미지 누락: {len(train_missing_images)}개")
    if val_missing_labels:
        print(f"⚠️  검증 세트 라벨 누락: {len(val_missing_labels)}개")
    if val_missing_images:
        print(f"⚠️  검증 세트 이미지 누락: {len(val_missing_images)}개")
    
    return True

def check_existing_models():
    """기존에 다운로드된 모델들과 학습된 모델들을 확인"""
    print("=== 기존 모델 확인 ===")
    
    # 사전 훈련된 모델들 확인
    pretrained_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']
    for model in pretrained_models:
        if os.path.exists(model):
            size = os.path.getsize(model) / (1024*1024)  # MB
            print(f"✅ {model} 존재 ({size:.1f}MB)")
        else:
            print(f"❌ {model} 없음")
    
    # 학습된 모델들 확인
    results_path = os.path.join(PROJECT_NAME, EXPERIMENT_NAME)
    if os.path.exists(results_path):
        weights_path = os.path.join(results_path, 'weights')
        if os.path.exists(weights_path):
            print(f"\n학습 결과 폴더: {results_path}")
            weight_files = glob.glob(os.path.join(weights_path, '*.pt'))
            for weight_file in weight_files:
                size = os.path.getsize(weight_file) / (1024*1024)
                print(f"  📁 {os.path.basename(weight_file)} ({size:.1f}MB)")
            
            # 체크포인트 파일들 확인
            checkpoints = glob.glob(os.path.join(weights_path, 'epoch*.pt'))
            if checkpoints:
                print(f"  💾 체크포인트 파일 수: {len(checkpoints)}개")
                
            # args.yaml 파일 확인 (원래 학습 설정)
            args_file = os.path.join(results_path, 'args.yaml')
            if os.path.exists(args_file):
                print(f"  ⚙️  원래 학습 설정 파일 존재: args.yaml")
                try:
                    with open(args_file, 'r') as f:
                        args = yaml.safe_load(f)
                    print(f"     - 원래 데이터셋: {args.get('data')}")
                    print(f"     - 원래 에포크: {args.get('epochs')}")
                    print(f"     - 원래 배치 크기: {args.get('batch')}")
                except:
                    print(f"     (설정 파일 읽기 실패)")
    else:
        print(f"\n❌ 학습 결과 폴더 없음: {results_path}")

def resume_training(yaml_path, project_name, experiment_name, resume_from=None):
    """학습 재개 (데이터셋 검증 포함)"""
    
    # 데이터셋 검증
    if not validate_dataset_structure(DATASET_ROOT):
        print("❌ 데이터셋 구조 검증 실패")
        return None
    
    # YAML 파일 확인/생성
    if not os.path.exists(yaml_path):
        print(f"❌ YAML 파일이 없습니다: {yaml_path}")
        print("dataset.yaml 파일을 생성하거나 올바른 경로를 지정해주세요.")
        return None
    
    if resume_from is None:
        # 자동으로 마지막 학습 재개
        results_path = os.path.join(project_name, experiment_name)
        last_model = os.path.join(results_path, 'weights', 'last.pt')
        
        if os.path.exists(last_model):
            print(f"🔄 마지막 체크포인트에서 학습 재개: {last_model}")
            model = YOLO(last_model)
        else:
            print(f"❌ 마지막 체크포인트를 찾을 수 없습니다: {last_model}")
            print("새로운 학습을 시작합니다...")
            model = YOLO('yolo11l.pt')
    else:
        # 특정 체크포인트에서 재개
        if os.path.exists(resume_from):
            print(f"🔄 지정된 체크포인트에서 학습 재개: {resume_from}")
            model = YOLO(resume_from)
        else:
            print(f"❌ 지정된 체크포인트를 찾을 수 없습니다: {resume_from}")
            return None
    
    # 학습 재개 (resume=True가 핵심!)
    results = model.train(
        data=yaml_path,
        epochs=1000,                  # 총 목표 에포크 (이미 진행된 에포크 + 추가 에포크)
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        project=project_name,
        name=experiment_name,
        resume=True,                  # 🔥 학습 재개 옵션
        save=True,
        save_period=10,
        patience=20,
        amp=True,
        lr0=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    return results

def continue_training_from_best(yaml_path, project_name, experiment_name, additional_epochs=100):
    """최고 성능 모델에서 추가 학습 (데이터셋 검증 포함)"""
    
    # 데이터셋 검증
    if not validate_dataset_structure(DATASET_ROOT):
        print("❌ 데이터셋 구조 검증 실패")
        return None
    
    results_path = os.path.join(project_name, experiment_name)
    best_model = os.path.join(results_path, 'weights', 'best.pt')
    
    if not os.path.exists(best_model):
        print(f"❌ 최고 성능 모델을 찾을 수 없습니다: {best_model}")
        return None
    
    print(f"🚀 최고 성능 모델에서 추가 학습 시작: {best_model}")
    
    # 새로운 실험 이름으로 추가 학습
    new_experiment_name = f"{experiment_name}_continued"
    
    model = YOLO(best_model)
    
    results = model.train(
        data=yaml_path,
        epochs=additional_epochs,     # 추가로 학습할 에포크 수
        imgsz=640,
        batch=16,
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        project=project_name,
        name=new_experiment_name,
        resume=False,                 # 새로운 학습이므로 False
        save=True,
        save_period=10,
        patience=20,
        amp=True,
        # 추가 학습시에는 더 낮은 학습률 사용
        lr0=0.001,                    # 원래보다 낮은 학습률
        weight_decay=0.0005,
        warmup_epochs=1,              # 더 짧은 warmup
        # 데이터 증강도 줄임
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,                   # 모자이크 증강 줄임
        mixup=0.0,
    )
    
    return results

def safe_resume_with_setup():
    """안전한 학습 재개 (모든 설정 자동 처리)"""
    
    print("=== 안전한 YOLO 학습 재개 ===\n")
    
    # 1. 데이터셋 구조 검증
    if not validate_dataset_structure(DATASET_ROOT):
        print("데이터셋 구조를 먼저 수정해주세요.")
        return None
    
    # 2. dataset.yaml 생성/확인
    yaml_path = create_or_check_dataset_yaml(DATASET_ROOT, CLASS_NAMES)
    
    # 3. 기존 모델들 확인
    check_existing_models()
    
    # 4. 사용자 선택에 따른 실행
    print(f"\n=== 실행 옵션 ===")
    print("1. 중단된 학습 재개 (last.pt에서)")
    print("2. 최고 성능 모델에서 추가 학습 (best.pt에서)")
    print("3. 새로 시작")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == "1":
        return resume_training(yaml_path, PROJECT_NAME, EXPERIMENT_NAME)
    elif choice == "2":
        epochs = input("추가 학습할 에포크 수 (기본 100): ").strip()
        epochs = int(epochs) if epochs.isdigit() else 100
        return continue_training_from_best(yaml_path, PROJECT_NAME, EXPERIMENT_NAME, epochs)
    elif choice == "3":
        model = YOLO('yolo11l.pt')
        return model.train(
            data=yaml_path,
            epochs=500,
            imgsz=640,
            batch=16,
            device='0' if torch.cuda.is_available() else 'cpu',
            project=PROJECT_NAME,
            name=f"{EXPERIMENT_NAME}_new",
            resume=False
        )
    else:
        print("올바른 선택지를 입력해주세요.")
        return None

def list_all_checkpoints(project_name, experiment_name):
    """모든 체크포인트 리스트 출력"""
    results_path = os.path.join(project_name, experiment_name)
    weights_path = os.path.join(results_path, 'weights')
    
    if not os.path.exists(weights_path):
        print(f"❌ 가중치 폴더가 없습니다: {weights_path}")
        return []
    
    checkpoints = []
    
    # 기본 모델들
    for model_name in ['best.pt', 'last.pt']:
        model_path = os.path.join(weights_path, model_name)
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)
            checkpoints.append((model_path, model_name, size))
    
    # 에포크별 체크포인트들
    epoch_checkpoints = glob.glob(os.path.join(weights_path, 'epoch*.pt'))
    for checkpoint in sorted(epoch_checkpoints):
        name = os.path.basename(checkpoint)
        size = os.path.getsize(checkpoint) / (1024*1024)
        checkpoints.append((checkpoint, name, size))
    
    print(f"=== 사용 가능한 체크포인트 ({len(checkpoints)}개) ===")
    for i, (path, name, size) in enumerate(checkpoints):
        print(f"{i+1:2d}. {name:<15} ({size:6.1f}MB) - {path}")
    
    return checkpoints

# 메인 실행 부분
if __name__ == "__main__":
    # 안전한 학습 재개 실행
    results = safe_resume_with_setup()
    
    if results:
        print("\n✅ 학습이 완료되었습니다!")
    else:
        print("\n❌ 학습 실행에 실패했습니다.")