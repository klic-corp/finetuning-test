"""
CodeLlama 7B Fine-tuning Script for HTML/CSS/JS Code
HTML, CSS, JavaScript 코드를 학습시키기 위한 CodeLlama 파인튜닝 프로그램
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import logging
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """파인튜닝 설정"""
    # 모델 설정
    model_name: str = "codellama/CodeLlama-7b-hf"
    
    # 데이터 경로
    data_dir: str = "./training_data"
    output_dir: str = "./codellama_finetuned"
    
    # LoRA 설정
    lora_r: int = 8  # 16 → 8 (메모리 절약)
    lora_alpha: int = 16  # 32 → 16
    lora_dropout: float = 0.05
    
    # 학습 설정
    batch_size: int = 1  # 4 → 1 (메모리 절약)
    gradient_accumulation_steps: int = 16  # 4 → 16 (효과적 배치 크기 유지)
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 1024  # 2048 → 1024 (메모리 절약)
    warmup_steps: int = 100
    
    # 양자화 설정
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    
    # 기타
    save_steps: int = 100
    logging_steps: int = 10
    seed: int = 42


class CodeDataset(Dataset):
    """HTML, CSS, JS 코드를 위한 데이터셋"""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        file_extensions: tuple = ('.html', '.css', '.js')
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        logger.info(f"데이터 로딩 중: {data_dir}")
        self._load_code_files(data_dir, file_extensions)
        logger.info(f"총 {len(self.data)}개의 코드 파일 로드 완료")
    
    def _load_code_files(self, data_dir: str, extensions: tuple):
        """코드 파일들을 로드"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise ValueError(f"데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        
        # 모든 파일 검색
        for ext in extensions:
            files = list(data_path.rglob(f"*{ext}"))
            logger.info(f"{ext} 파일 {len(files)}개 발견")
            
            for file_path in tqdm(files, desc=f"{ext} 파일 로딩"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if content.strip():  # 빈 파일 제외
                        file_type = ext[1:].upper()  # .html -> HTML
                        
                        # 프롬프트 형식으로 구성
                        prompt = self._create_training_prompt(
                            file_type=file_type,
                            code=content,
                            filename=file_path.name
                        )
                        
                        self.data.append({
                            'text': prompt,
                            'file_path': str(file_path),
                            'file_type': file_type
                        })
                        
                except Exception as e:
                    logger.warning(f"파일 로드 실패 {file_path}: {e}")
    
    def _create_training_prompt(
        self,
        file_type: str,
        code: str,
        filename: str
    ) -> str:
        """학습용 프롬프트 생성"""
        # CodeLlama를 위한 instruction 형식
        prompt = f"""### Instruction:
Write a {file_type} code for {filename}.

### Input:
Create {file_type} code following best practices.

### Response:
```{file_type.lower()}
{code}
```

"""
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 토크나이징
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class CodeLlamaFineTuner:
    """CodeLlama 파인튜닝 클래스"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")
        
        # CUDA 메모리 최적화 설정
        if torch.cuda.is_available():
            # 메모리 조각화 방지
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            # CUDA 캐시 정리
            torch.cuda.empty_cache()
            # 메모리 상태 출력
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"총 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 시드 설정
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def load_model_and_tokenizer(self):
        """모델과 토크나이저 로드"""
        logger.info(f"모델 로딩: {self.config.model_name}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # 양자화 설정 (메모리 효율성)
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 모델 로드
        logger.info("모델 다운로드 및 로딩 중... (시간이 걸릴 수 있습니다)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,  # CPU 메모리 사용 최적화
            max_memory={0: "13GB"},  # GPU 메모리 제한 설정
        )
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # LoRA 설정
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # use_cache 비활성화 (gradient checkpointing과 호환 불가)
        self.model.config.use_cache = False
        
        logger.info("모델 및 토크나이저 로드 완료")
    
    def prepare_datasets(self):
        """데이터셋 준비"""
        logger.info("데이터셋 준비 중...")
        
        # 학습 데이터 로드
        train_dataset = CodeDataset(
            data_dir=self.config.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        return train_dataset
    
    def train(self):
        """모델 학습"""
        logger.info("=" * 50)
        logger.info("CodeLlama 7B 파인튜닝 시작")
        logger.info("=" * 50)
        
        # 모델과 토크나이저 로드
        logger.info("1/4: 모델 로딩 중...")
        self.load_model_and_tokenizer()
        
        # 데이터셋 준비
        logger.info("2/4: 데이터셋 준비 중...")
        train_dataset = self.prepare_datasets()
        logger.info(f"데이터셋 크기: {len(train_dataset)} 샘플")
        
        # 학습 설정
        logger.info("3/4: 학습 설정 구성 중...")
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),  # CUDA 사용 가능할 때만 fp16 활성화
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            warmup_steps=self.config.warmup_steps,
            save_total_limit=3,
            report_to="none",
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            group_by_length=False,  # 초기 지연 방지를 위해 비활성화 (필요시 True로 변경)
            dataloader_pin_memory=torch.cuda.is_available(),  # CUDA 사용 가능할 때만 pin_memory 활성화
            gradient_checkpointing=True,  # 메모리 절약을 위한 gradient checkpointing 명시적 활성화
            gradient_checkpointing_kwargs={"use_reentrant": False},  # PyTorch 2.9 호환성
            disable_tqdm=False,  # tqdm 진행바 활성화
            logging_first_step=True,  # 첫 스텝부터 로깅
        )
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 트레이너 설정
        logger.info("4/4: 트레이너 초기화 중...")
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # 총 스텝 계산 및 출력
        total_steps = (len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)) * self.config.num_epochs
        logger.info(f"총 학습 스텝: {total_steps}")
        logger.info(f"에폭당 스텝: {len(train_dataset) // (self.config.batch_size * self.config.gradient_accumulation_steps)}")
        
        # 학습 시작
        logger.info("=" * 50)
        logger.info("학습 시작... (첫 번째 배치는 시간이 걸릴 수 있습니다)")
        logger.info("=" * 50)
        self.trainer.train()
        
        # 모델 저장
        logger.info(f"모델 저장 중: {self.config.output_dir}")
        
        # LoRA 어댑터만 저장 (더 효율적)
        self.model.save_pretrained(
            self.config.output_dir,
            safe_serialization=True  # safetensors 형식으로 저장
        )
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        logger.info("=" * 50)
        logger.info("파인튜닝 완료!")
        logger.info("=" * 50)
    
    def test_model(self, prompt: str):
        """학습된 모델 테스트"""
        logger.info("모델 테스트 중...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result


def main():
    """메인 함수"""
    # GPU 메모리 상태 확인
    if torch.cuda.is_available():
        logger.info("=" * 50)
        logger.info("GPU 메모리 상태")
        logger.info("=" * 50)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"총 VRAM: {total_memory:.2f} GB")
        
        # 14GB 이하면 자동으로 최적화된 설정 사용
        if total_memory < 16:
            logger.warning(f"⚠️  VRAM이 16GB 미만입니다. 메모리 절약 모드로 설정합니다.")
            logger.warning("   batch_size=1, max_seq_length=1024로 자동 조정됨")
    
    # 설정 (기본값이 이미 메모리 절약 모드로 변경됨)
    config = FineTuningConfig(
        data_dir="./training_data",  # HTML/CSS/JS 파일들이 있는 디렉토리
        output_dir="./codellama_7b_finetuned",
        # batch_size=1,  # 기본값 사용 (메모리 절약)
        # num_epochs=3,  # 기본값 사용
        # max_seq_length=1024,  # 기본값 사용 (메모리 절약)
    )
    
    # 파인튜너 생성 및 학습
    fine_tuner = CodeLlamaFineTuner(config)
    fine_tuner.train()
    
    # 테스트 예시
    test_prompt = """### Instruction:
Write a HTML code for responsive navigation bar.

### Input:
Create HTML code following best practices.

### Response:
"""
    
    logger.info("\n" + "=" * 50)
    logger.info("테스트 생성 결과:")
    logger.info("=" * 50)
    result = fine_tuner.test_model(test_prompt)
    print(result)


if __name__ == "__main__":
    main()
