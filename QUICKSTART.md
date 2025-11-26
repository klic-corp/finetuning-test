# CodeLlama 7B Fine-tuning 빠른 시작 가이드

## 1단계: 환경 설정

### Python 가상환경 생성 (권장)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 필수 라이브러리 설치
```powershell
pip install -r requirements.txt
```

### CUDA 확인
```powershell
python -c "import torch; print(f'CUDA 사용 가능: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"없음\"}')"
```

## 2단계: 데이터 준비

### 데이터 디렉토리 생성
```powershell
New-Item -ItemType Directory -Path "training_data" -Force
```

### 데이터 구조
학습 데이터를 `training_data` 폴더에 넣으세요:
```
training_data/
├── *.html
├── *.css
└── *.js
```

또는 프로젝트 단위로:
```
training_data/
├── project1/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── project2/
│   └── ...
```

### 데이터 검증
```powershell
python preprocess_data.py --data_dir training_data --validate
```

### 데이터 통계 확인
```powershell
python preprocess_data.py --data_dir training_data
```

## 3단계: 모델 파인튜닝

### 기본 학습 시작
```powershell
python tuning.py
```

### 커스텀 설정으로 학습
`tuning.py`의 `main()` 함수에서 설정 수정 후 실행:
```python
config = FineTuningConfig(
    data_dir="./training_data",
    output_dir="./my_model",
    batch_size=2,  # GPU 메모리에 맞게 조정
    num_epochs=3,
)
```

## 4단계: 학습된 모델 사용

### CLI 모드
```powershell
python generate_code.py --instruction "Create a responsive navigation bar" --code_type HTML
```

### 대화형 모드
```powershell
python generate_code.py
```

### Python 스크립트에서 사용
```python
from generate_code import CodeGenerator

generator = CodeGenerator("./codellama_7b_finetuned")
code = generator.generate_code(
    instruction="Create a contact form",
    code_type="HTML"
)
print(code)
```

## GPU 메모리별 권장 설정

### 12GB VRAM (RTX 3060, RTX 3080Ti 등)
```python
config = FineTuningConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
    max_seq_length=1024,
)
```

### 16GB VRAM (RTX 4060Ti, RTX A4000 등)
```python
config = FineTuningConfig(
    batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=1536,
)
```

### 24GB VRAM (RTX 3090, RTX 4090 등)
```python
config = FineTuningConfig(
    batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=2048,
)
```

## 문제 해결

### CUDA Out of Memory 오류
1. `batch_size` 감소 (예: 4 → 2 → 1)
2. `max_seq_length` 감소 (예: 2048 → 1024)
3. `gradient_accumulation_steps` 증가

### 학습이 시작되지 않음
1. CUDA 설치 확인
2. PyTorch CUDA 버전 확인
3. 드라이버 업데이트

### 모델 다운로드 실패
1. 인터넷 연결 확인
2. Hugging Face 토큰 설정 (필요시)
3. 수동 다운로드 후 로컬 경로 지정

## 유용한 명령어

### 학습 진행 상황 모니터링
학습 중 다른 터미널에서:
```powershell
# GPU 사용률 확인
nvidia-smi

# 실시간 모니터링
nvidia-smi -l 1
```

### 체크포인트에서 재개
학습이 중단된 경우, `output_dir`의 마지막 체크포인트에서 자동 재개됩니다.

### 모델 크기 확인
```powershell
Get-ChildItem -Recurse ./codellama_7b_finetuned | Measure-Object -Property Length -Sum
```

## 다음 단계

1. **모델 평가**: 생성된 코드 품질 평가
2. **하이퍼파라미터 튜닝**: 더 나은 성능을 위한 파라미터 조정
3. **데이터 증강**: 더 많은 학습 데이터 추가
4. **배포**: 모델을 API나 웹 서비스로 배포

## 참고 링크

- [README.md](README.md) - 자세한 문서
- [tuning.py](tuning.py) - 학습 스크립트
- [generate_code.py](generate_code.py) - 코드 생성 스크립트
- [preprocess_data.py](preprocess_data.py) - 데이터 전처리 도구
