# CodeLlama 7B 파인튜닝 시스템 요구사항

## 🖥️ 하드웨어 사양

### ✅ 최소 사양 (4-bit 양자화 사용 시)

#### GPU
- **NVIDIA GPU**: 최소 **12GB VRAM**
  - RTX 3060 (12GB) - 최소 사양
  - RTX 3060Ti (8GB) - ❌ **메모리 부족**
  - RTX 4060 (8GB) - ❌ **메모리 부족**

#### CPU & RAM
- **CPU**: Intel i5/AMD Ryzen 5 이상 (8코어 권장)
- **RAM**: **최소 16GB**, **32GB 강력 권장**
  - 16GB: 가능하지만 다른 프로그램 종료 필수
  - 32GB: 안정적 운영 가능

#### 저장 공간
- **최소 50GB 여유 공간**
  - 모델 다운로드: ~13GB (CodeLlama 7B)
  - 양자화 모델: ~4GB
  - 체크포인트 저장: ~10-20GB
  - 학습 데이터: 데이터 크기에 따라 다름

---

### 🚀 권장 사양 (효율적 학습)

#### GPU
- **NVIDIA GPU**: **16GB VRAM 이상**
  - ✅ RTX 4060Ti (16GB)
  - ✅ RTX 4070 (12GB) - 가능하지만 여유 적음
  - ✅ RTX A4000 (16GB)
  - ✅ RTX 3080 (10/12GB) - 배치 크기 조정 필요

#### CPU & RAM
- **CPU**: Intel i7/AMD Ryzen 7 이상 (12코어 이상)
- **RAM**: **32GB**

#### 저장 공간
- **100GB 이상 여유 공간**

---

### 🔥 최적 사양 (빠른 학습)

#### GPU
- **NVIDIA GPU**: **24GB VRAM**
  - ⭐ RTX 3090 (24GB)
  - ⭐ RTX 4090 (24GB)
  - ⭐ RTX A5000 (24GB)
  - ⭐ A100 (40GB/80GB)

#### CPU & RAM
- **CPU**: Intel i9/AMD Ryzen 9 이상 (16코어 이상)
- **RAM**: **64GB**

#### 저장 공간
- **200GB 이상 (SSD NVMe 권장)**

---

## 💾 현재 코드 설정별 메모리 사용량

### 기본 설정 (tuning.py)
```python
batch_size = 4
gradient_accumulation_steps = 4
max_seq_length = 2048
use_4bit = True  # 4-bit 양자화 활성화
```

**예상 VRAM 사용량**: ~10-12GB

### 설정별 VRAM 사용량 비교

| 설정 | VRAM | 권장 GPU |
|------|------|----------|
| batch_size=1, max_seq_length=1024 | ~6-8GB | RTX 3060 (12GB) |
| batch_size=2, max_seq_length=1024 | ~8-10GB | RTX 3060 (12GB) |
| batch_size=4, max_seq_length=2048 | ~10-12GB | RTX 3060 (12GB), RTX 4060Ti (16GB) |
| batch_size=8, max_seq_length=2048 | ~18-20GB | RTX 3090 (24GB), RTX 4090 (24GB) |

---

## 🔧 소프트웨어 요구사항

### 필수
- **Python**: 3.10 이상 (3.11 권장)
- **CUDA**: 11.8 이상 (12.1 권장)
- **cuDNN**: CUDA 버전과 호환
- **PyTorch**: 2.0 이상 (CUDA 지원 버전)

### 설치 확인
```powershell
# Python 버전
python --version

# CUDA 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'Version: {torch.version.cuda}')"
```

---

## 📊 GPU별 권장 설정

### RTX 3060 (12GB)
```python
config = FineTuningConfig(
    batch_size=1,
    gradient_accumulation_steps=16,
    max_seq_length=1024,
    use_4bit=True,
)
```
- **학습 시간**: 10,000 파일 기준 약 **8-12시간**

### RTX 4060Ti (16GB)
```python
config = FineTuningConfig(
    batch_size=2,
    gradient_accumulation_steps=8,
    max_seq_length=1536,
    use_4bit=True,
)
```
- **학습 시간**: 10,000 파일 기준 약 **6-8시간**

### RTX 3090/4090 (24GB)
```python
config = FineTuningConfig(
    batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=2048,
    use_4bit=True,
)
```
- **학습 시간**: 10,000 파일 기준 약 **3-5시간**

### A100 (40GB/80GB)
```python
config = FineTuningConfig(
    batch_size=8,
    gradient_accumulation_steps=2,
    max_seq_length=2048,
    use_4bit=False,  # 양자화 불필요
)
```
- **학습 시간**: 10,000 파일 기준 약 **2-3시간**

---

## ⚠️ GPU 없이 CPU만으로 학습 가능한가?

**불가능**은 아니지만 **비현실적**입니다:
- CodeLlama 7B는 CPU 학습 시 **수백 배 느림**
- 예상 시간: 10,000 파일 기준 **수 주~수 개월**
- **권장하지 않음**

CPU 전용 대안:
1. **Google Colab** (무료 GPU 제공)
2. **Kaggle Notebooks** (무료 GPU 제공)
3. **AWS/Azure/GCP** 클라우드 GPU 인스턴스 대여

---

## 🌐 클라우드 옵션 (GPU 없는 경우)

### Google Colab
- **무료**: T4 GPU (16GB) 제공
- **Pro**: A100 GPU 사용 가능
- **비용**: 무료 또는 월 $9.99

### Kaggle
- **무료**: P100 GPU (16GB) 또는 T4 GPU
- **제한**: 주당 30시간

### AWS EC2
- **g4dn.xlarge**: T4 GPU (16GB) - 시간당 ~$0.5
- **p3.2xlarge**: V100 GPU (16GB) - 시간당 ~$3

### Paperspace Gradient
- **무료 티어**: 제한적 GPU 사용
- **Pro**: 다양한 GPU 옵션

---

## 📈 메모리 부족 시 해결책

### 1. 배치 크기 감소
```python
batch_size = 1  # 4 → 1
gradient_accumulation_steps = 16  # 4 → 16
```

### 2. 시퀀스 길이 감소
```python
max_seq_length = 1024  # 2048 → 1024
```

### 3. LoRA Rank 감소
```python
lora_r = 8  # 16 → 8
```

### 4. Gradient Checkpointing 확인
```python
gradient_checkpointing = True  # 이미 활성화됨
```

---

## 🎯 요약

### ✅ 실행 가능한 최소 사양
- **GPU**: RTX 3060 (12GB) 이상
- **RAM**: 16GB 이상 (32GB 권장)
- **저장공간**: 50GB 이상
- **CUDA**: 11.8 이상

### ❌ 실행 불가능
- **8GB VRAM GPU** (RTX 3060Ti 8GB, RTX 4060 8GB 등)
- **CPU 전용** (비현실적으로 느림)
- **16GB 미만 RAM**

### 💡 권장 구성
- **GPU**: RTX 4090 (24GB) 또는 RTX 3090 (24GB)
- **RAM**: 32GB
- **저장공간**: 100GB SSD
- **CUDA**: 12.1 이상

---

## 📞 GPU가 없다면?

1. **Google Colab** 사용 (무료 T4 GPU)
2. **Kaggle Notebooks** 사용 (무료 P100/T4)
3. **클라우드 GPU** 임대 (AWS, GCP, Azure)
4. **로컬 GPU** 구입 (RTX 3060 12GB 이상)

---

## ⏱️ 예상 학습 시간

| 데이터 크기 | RTX 3060 (12GB) | RTX 4090 (24GB) |
|-------------|-----------------|-----------------|
| 1,000 파일 | 1-2시간 | 20-30분 |
| 10,000 파일 | 8-12시간 | 3-5시간 |
| 50,000 파일 | 2-3일 | 12-20시간 |
| 100,000 파일 | 4-6일 | 1-2일 |

**참고**: 실제 시간은 파일 크기, 코드 복잡도, 시스템 성능에 따라 달라질 수 있습니다.
