# UI모아 Fine-tuning Guide

UI 모아의 HTML, CSS, JavaScript 코드를 학습시키기 위한 CodeLlama 7B 파인튜닝 프로그램입니다.

## 주요 기능

- **대용량 데이터 처리**: HTML, CSS, JS 파일을 자동으로 스캔하고 로드
- **LoRA 최적화**: 메모리 효율적인 파인튜닝을 위한 LoRA (Low-Rank Adaptation) 사용
- **4-bit 양자화**: GPU 메모리 사용량 감소를 위한 양자화 지원
- **자동 프롬프트 생성**: 학습 데이터를 instruction 형식으로 자동 변환

## 필수 요구사항

### 소프트웨어
- Python 3.10 이상
- CUDA 11.8 이상
- PyTorch 2.0 이상

## 설치 방법

1. **저장소 클론 및 의존성 설치**

```bash
pip install -r requirements.txt
```

2. **CUDA 확인**

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 데이터 준비

### 디렉토리 구조

학습 데이터를 다음과 같은 구조로 배치합니다.

```
training_data/
├── html/
│   ├── page1.html
│   ├── page2.html
│   └── ...
├── css/
│   ├── style1.css
│   ├── style2.css
│   └── ...
└── js/
    ├── script1.js
    ├── script2.js
    └── ...
```

또는 하위 폴더에 섞여 있어도 자동으로 스캔합니다:

```
training_data/
├── project1/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── project2/
│   ├── main.html
│   ├── theme.css
│   └── logic.js
└── ...
```

**지원하는 파일 형식:**
- HTML: `.html`, `.htm`
- CSS: `.css`
- JavaScript: `.js`, `.jsx`
- TypeScript: `.ts`, `.tsx`

## 사용 방법

### 1. 기본 학습

```bash
python tuning.py
```

### 2. 설정 커스터마이징

`tuning.py`의 `main()` 함수에서 설정을 할 수 있습니다.

```python
config = FineTuningConfig(
    data_dir="./training_data",           # 학습 데이터 경로
    output_dir="./codellama_7b_finetuned", # 모델 저장 경로
    
    # 학습 파라미터
    batch_size=4,                          # GPU 메모리에 따라 조정 (1-8)
    gradient_accumulation_steps=4,         # 효과적 배치 크기 = batch_size * 이 값
    num_epochs=3,                          # 학습 에폭 수
    learning_rate=2e-4,                    # 학습률
    max_seq_length=2048,                   # 최대 시퀀스 길이
    
    # LoRA 파라미터
    lora_r=16,                             # LoRA rank
    lora_alpha=32,                         # LoRA alpha
    lora_dropout=0.05,                     # LoRA dropout
    
    # 기타
    save_steps=100,                        # 체크포인트 저장 주기
    logging_steps=10,                      # 로그 출력 주기
)
```

### 3. 메모리 부족 시 설정

GPU 메모리가 부족한 경우:

```python
config = FineTuningConfig(
    batch_size=1,                          # 배치 크기 감소
    gradient_accumulation_steps=16,        # 그래디언트 누적 증가
    max_seq_length=1024,                   # 시퀀스 길이 감소
)
```

## 학습 과정

프로그램은 다음 단계로 진행됩니다:

1. **모델 로드**: CodeLlama 7B 모델 다운로드 및 로드
2. **양자화**: 4-bit 양자화로 메모리 효율성 향상
3. **LoRA 적용**: 효율적인 파인튜닝을 위한 LoRA 설정
4. **데이터 로드**: HTML/CSS/JS 파일 스캔 및 로드
5. **프롬프트 생성**: Instruction 형식으로 데이터 변환
6. **학습**: 모델 파인튜닝 실행
7. **저장**: 학습된 모델 저장

## 학습 후 모델 사용

### Python에서 사용

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 베이스 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "codellama/CodeLlama-7b-hf",
    device_map="auto",
    torch_dtype=torch.float16
)

# 파인튜닝된 모델 로드
model = PeftModel.from_pretrained(base_model, "./codellama_7b_finetuned")
tokenizer = AutoTokenizer.from_pretrained("./codellama_7b_finetuned")

# 코드 생성
prompt = """### Instruction:
이름, 연락처, 이메일, 문의사항이 있는 폼을 만들어줘.

### Input:
Create HTML code following best practices.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 하이퍼파라미터 설명

### LoRA 파라미터
- **lora_r**: LoRA의 rank (낮을수록 메모리↓, 성능↓) [8, 16, 32]
- **lora_alpha**: LoRA scaling factor (보통 lora_r의 2배)
- **lora_dropout**: 과적합 방지를 위한 드롭아웃 [0.05-0.1]

### 학습 파라미터
- **batch_size**: 한 번에 처리할 샘플 수 (GPU 메모리에 따라)
- **gradient_accumulation_steps**: 그래디언트 누적 단계
- **learning_rate**: 학습률 [1e-5 ~ 5e-4]
- **num_epochs**: 전체 데이터를 학습할 횟수 [3-5]
- **max_seq_length**: 최대 토큰 길이 [512, 1024, 2048]

## 문제 해결

### CUDA Out of Memory
```python
# 배치 크기 감소
batch_size=1
gradient_accumulation_steps=16
max_seq_length=1024
```

### 학습이 너무 느림
```python
# 데이터 샘플링
# CodeDataset 클래스에서 일부 데이터만 사용
```

### 모델 성능이 낮음
- 학습 데이터 품질 확인
- 에폭 수 증가
- Learning rate 조정
- LoRA rank 증가

## 참고 자료

- [CodeLlama Paper](https://arxiv.org/abs/2308.12950)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
