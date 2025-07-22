# YOLO v11 학습 재개 및 추가 학습

import torch
import os
from ultralytics import YOLO
import glob

# 기본 설정
DATASET_ROOT = "./dataset"
PROJECT_NAME = "beekeeping"
EXPERIMENT_NAME = "dreambee"
CLASS_NAMES = ['egg', 'lava', 'pupa', 'bee', 'queen']

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
    else:
        print(f"\n❌ 학습 결과 폴더 없음: {results_path}")

def resume_training(yaml_path, project_name, experiment_name, resume_from=None):
    """학습 재개"""
    
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
    """최고 성능 모델에서 추가 학습"""
    
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

def transfer_learning_from_checkpoint(yaml_path, project_name, experiment_name, checkpoint_path, new_experiment_name):
    """특정 체크포인트에서 새로운 설정으로 전이 학습"""
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 체크포인트를 찾을 수 없습니다: {checkpoint_path}")
        return None
    
    print(f"🔄 체크포인트에서 전이 학습 시작: {checkpoint_path}")
    
    model = YOLO(checkpoint_path)
    
    results = model.train(
        data=yaml_path,
        epochs=500,                   # 새로운 에포크 설정
        imgsz=640,
        batch=8,                      # 다른 배치 크기
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,
        project=project_name,
        name=new_experiment_name,
        resume=False,                 # 새로운 실험이므로 False
        save=True,
        save_period=5,                # 더 자주 저장
        patience=30,
        amp=True,
        lr0=0.005,                    # 다른 학습률
        weight_decay=0.001,
        warmup_epochs=5,
        # 다른 증강 설정
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=5.0,
        translate=0.2,
        scale=0.8,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    return results

def smart_model_loader(preferred_model='yolo11l.pt'):
    """똑똑한 모델 로더: 기존 파일이 있으면 사용, 없으면 다운로드"""
    
    if os.path.exists(preferred_model):
        size = os.path.getsize(preferred_model) / (1024*1024)
        print(f"✅ 기존 모델 사용: {preferred_model} ({size:.1f}MB)")
        return YOLO(preferred_model)
    else:
        print(f"📥 모델 다운로드 중: {preferred_model}")
        return YOLO(preferred_model)  # 자동으로 다운로드됨

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

# 사용 예시 함수들
def example_resume_interrupted_training():
    """중단된 학습 재개 예시"""
    yaml_path = "dataset.yaml"
    
    print("중단된 학습을 재개합니다...")
    results = resume_training(yaml_path, PROJECT_NAME, EXPERIMENT_NAME)
    return results

def example_continue_from_best():
    """최고 성능 모델에서 추가 학습 예시"""
    yaml_path = "dataset.yaml"
    
    print("최고 성능 모델에서 100 에포크 추가 학습...")
    results = continue_training_from_best(yaml_path, PROJECT_NAME, EXPERIMENT_NAME, 100)
    return results

def example_transfer_learning():
    """전이 학습 예시"""
    yaml_path = "dataset.yaml"
    
    # 먼저 사용 가능한 체크포인트 확인
    checkpoints = list_all_checkpoints(PROJECT_NAME, EXPERIMENT_NAME)
    
    if checkpoints:
        # 예: best.pt에서 전이 학습
        best_checkpoint = None
        for path, name, size in checkpoints:
            if name == 'best.pt':
                best_checkpoint = path
                break
        
        if best_checkpoint:
            results = transfer_learning_from_checkpoint(
                yaml_path, 
                PROJECT_NAME, 
                EXPERIMENT_NAME, 
                best_checkpoint, 
                "dreambee_transfer"
            )
            return results
    
    print("사용 가능한 체크포인트가 없습니다.")
    return None

# 메인 실행 부분
if __name__ == "__main__":
    print("=== YOLO 학습 재개/추가 학습 도구 ===\n")
    
    # 1. 기존 모델들 확인
    check_existing_models()
    
    # 2. 사용 가능한 체크포인트 확인
    print(f"\n=== 체크포인트 확인 ===")
    checkpoints = list_all_checkpoints(PROJECT_NAME, EXPERIMENT_NAME)
    
    # 3. 선택에 따라 실행
    print(f"\n=== 실행 옵션 ===")
    print("1. 중단된 학습 재개 (resume_training)")
    print("2. 최고 성능 모델에서 추가 학습 (continue_training_from_best)")
    print("3. 전이 학습 (transfer_learning_from_checkpoint)")
    print("4. 처음부터 새로 시작 (smart_model_loader 사용)")
    
    # 예시 실행 (실제로는 선택해서 실행)
    choice = input("\n선택하세요 (1-4): ").strip()
    
    if choice == "1":
        example_resume_interrupted_training()
    elif choice == "2":
        example_continue_from_best()
    elif choice == "3":
        example_transfer_learning()
    elif choice == "4":
        # 새로 시작하지만 기존 모델 파일 재사용
        model = smart_model_loader('yolo11l.pt')
        print("새로운 학습을 시작할 수 있습니다.")
    else:
        print("올바른 선택지를 입력해주세요.")