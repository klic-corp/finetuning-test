"""
학습된 CodeLlama 모델을 사용하여 코드를 생성하는 예제 스크립트
HTML, CSS, JS 파일을 생성하고 result 폴더에 저장하며 스크린샷 생성
"""

import os
import re
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse


class CodeGenerator:
    """코드 생성기 클래스"""
    
    def __init__(self, model_path: str, base_model: str = "codellama/CodeLlama-7b-hf"):
        """
        Args:
            model_path: 파인튜닝된 모델 경로
            base_model: 베이스 모델 이름
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"디바이스: {self.device}")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"총 VRAM: {total_memory:.2f} GB")
        
        print("모델 로딩 중...")
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 4-bit 양자화 설정 (추론 시 메모리 절약)
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # 베이스 모델 로드 (4-bit 양자화)
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "13GB"},  # GPU 메모리 제한
                offload_folder="./offload_tmp",  # 오프로드 디렉토리
            )
        else:
            # CPU 모드
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="cpu",
                dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        
        # 파인튜닝된 가중치 로드
        self.model = PeftModel.from_pretrained(
            base, 
            model_path,
            is_trainable=False  # 추론 모드
        )
        self.model.eval()
        print("모델 로드 완료!")
        
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_code(
        self,
        instruction: str,
        code_type: str = "HTML",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95
    ) -> str:
        """
        코드 생성
        
        Args:
            instruction: 생성할 코드에 대한 설명
            code_type: 코드 타입 (HTML, CSS, JS)
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (낮을수록 보수적)
            top_p: nucleus sampling 파라미터
        
        Returns:
            생성된 코드
        """
        prompt = f"""### Instruction:
{instruction}

### Input:
Create {code_type} code following best practices.

### Response:
```{code_type.lower()}
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # 반복 방지
                no_repeat_ngram_size=3,  # 3-gram 반복 방지
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 디버그: 생성된 토큰 수 확인
        generated_tokens = outputs[0].shape[0] - inputs['input_ids'].shape[1]
        if generated_tokens >= max_tokens - 10:
            print(f"⚠️  경고: max_tokens({max_tokens})에 도달했습니다. 코드가 잘렸을 수 있습니다.")
            print(f"   max_tokens를 늘려보세요 (예: --max_tokens {max_tokens * 2})")
        
        # Response 부분만 추출
        if "### Response:" in result:
            result = result.split("### Response:")[1].strip()
        
        # 코드 블록에서 코드만 추출
        result = self._extract_code(result, code_type)
        
        return result
    
    def _extract_code(self, text: str, code_type: str) -> str:
        """생성된 텍스트에서 코드만 추출"""
        # 코드 블록 추출 (```html ... ``` 형식)
        # 닫는 ``` 이 없어도 처리 가능하도록 수정
        pattern = rf"```(?:{code_type.lower()}|html|css|javascript|js)?\s*\n(.*?)(?:\n```|$)"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            code = matches[0].strip()
            # 코드가 너무 짧으면 (잘렸을 가능성) 경고
            if len(code) < 50:
                print(f"⚠️  경고: 생성된 코드가 매우 짧습니다 ({len(code)} 문자). max_tokens를 늘려보세요.")
            return code
        
        # 코드 블록 마커가 시작만 있고 끝이 없는 경우
        start_pattern = rf"```(?:{code_type.lower()}|html|css|javascript|js)?\s*\n(.*)"
        start_matches = re.findall(start_pattern, text, re.DOTALL | re.IGNORECASE)
        if start_matches:
            code = start_matches[0].strip()
            print(f"⚠️  경고: 코드 블록이 완전하게 닫히지 않았습니다. max_tokens를 늘려보세요.")
            return code
        
        # 코드 블록이 없으면 전체 텍스트 반환
        if len(text.strip()) < 100:
            print(f"⚠️  경고: 생성된 텍스트가 매우 짧습니다. max_tokens를 늘려보세요.")
        return text.strip()
    
    def save_code(self, code: str, code_type: str, filename: str, output_dir: str = "./result") -> str:
        """생성된 코드를 파일로 저장"""
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 파일 확장자 설정
        ext_map = {"HTML": ".html", "CSS": ".css", "JS": ".js"}
        ext = ext_map.get(code_type, ".txt")
        
        # 파일 저장
        file_path = output_path / f"{filename}{ext}"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        print(f"✓ 파일 저장 완료: {file_path}")
        return str(file_path)
    
    def generate_all_types(
        self,
        project_name: str,
        html_instruction: str = None,
        css_instruction: str = None,
        js_instruction: str = None,
        output_dir: str = "./result",
        max_tokens: int = 2048  # 기본값을 더 크게 설정
    ) -> dict:
        """HTML, CSS, JS 파일을 모두 생성"""
        results = {}
        
        # 기본 instruction 설정
        if not html_instruction:
            html_instruction = f"Create a complete HTML page for {project_name}"
        if not css_instruction:
            css_instruction = f"Create CSS styles for {project_name}"
        if not js_instruction:
            js_instruction = f"Create JavaScript functionality for {project_name}"
        
        print(f"\n{'='*70}")
        print(f"프로젝트: {project_name}")
        print(f"max_tokens: {max_tokens} (파일별)")
        print(f"{'='*70}\n")
        
        # HTML 생성
        print("1/3: HTML 생성 중...")
        html_code = self.generate_code(html_instruction, "HTML", max_tokens=max_tokens)
        html_path = self.save_code(html_code, "HTML", project_name, output_dir)
        results['html'] = {'code': html_code, 'path': html_path}
        
        # CSS 생성
        print("\n2/3: CSS 생성 중...")
        css_code = self.generate_code(css_instruction, "CSS", max_tokens=max_tokens)
        css_path = self.save_code(css_code, "CSS", project_name, output_dir)
        results['css'] = {'code': css_code, 'path': css_path}
        
        # JavaScript 생성
        print("\n3/3: JavaScript 생성 중...")
        js_code = self.generate_code(js_instruction, "JS", max_tokens=max_tokens)
        js_path = self.save_code(js_code, "JS", project_name, output_dir)
        results['js'] = {'code': js_code, 'path': js_path}
        
        print(f"\n{'='*70}")
        print("✓ 모든 파일 생성 완료!")
        print(f"{'='*70}\n")
        
        return results
    
    def create_screenshot(self, html_path: str, output_dir: str = "./result"):
        """HTML 파일의 스크린샷 생성"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            import time
            
            print("\n스크린샷 생성 중...")
            
            # Chrome 옵션 설정
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # 헤드리스 모드
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--window-size=1920,1080')
            
            # WebDriver 설정
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # HTML 파일 열기
            html_full_path = Path(html_path).absolute()
            driver.get(f'file:///{html_full_path}')
            
            # 페이지 로드 대기
            time.sleep(2)
            
            # 스크린샷 저장
            screenshot_path = Path(output_dir) / f"{Path(html_path).stem}_screenshot.png"
            driver.save_screenshot(str(screenshot_path))
            
            driver.quit()
            
            print(f"✓ 스크린샷 저장 완료: {screenshot_path}")
            return str(screenshot_path)
            
        except ImportError:
            print("⚠️  스크린샷 생성을 위해 selenium과 webdriver-manager 설치가 필요합니다:")
            print("   pip install selenium webdriver-manager")
            return None
        except Exception as e:
            print(f"⚠️  스크린샷 생성 실패: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="CodeLlama 코드 생성기 - HTML/CSS/JS 생성 및 저장")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./codellama_7b_finetuned",
        help="파인튜닝된 모델 경로"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="프로젝트 이름 (HTML/CSS/JS 모두 생성)"
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Write a responsive navigation bar with dropdown menu",
        help="생성할 코드에 대한 설명"
    )
    parser.add_argument(
        "--code_type",
        type=str,
        default="HTML",
        choices=["HTML", "CSS", "JS", "ALL"],
        help="생성할 코드 타입 (ALL: 모든 타입 생성)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result",
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="HTML 스크린샷 생성"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,  # 기본값을 1024에서 2048로 증가
        help="최대 생성 토큰 수 (기본값: 2048, 더 긴 코드는 4096 권장)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="생성 온도 (0.0-1.0)"
    )
    
    args = parser.parse_args()
    
    # 코드 생성기 초기화
    generator = CodeGenerator(args.model_path)
    
    # 모든 타입 생성 (ALL 또는 project_name 지정 시)
    if args.code_type == "ALL" or args.project_name:
        project_name = args.project_name or f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = generator.generate_all_types(
            project_name=project_name,
            html_instruction=args.instruction if args.code_type == "ALL" else None,
            output_dir=args.output_dir,
            max_tokens=args.max_tokens  # max_tokens 전달
        )
        
        # 스크린샷 생성
        if args.screenshot and results.get('html'):
            generator.create_screenshot(results['html']['path'], args.output_dir)
    
    # 단일 타입 생성
    else:
        print("\n" + "=" * 70)
        print(f"코드 타입: {args.code_type}")
        print(f"설명: {args.instruction}")
        print("=" * 70 + "\n")
        
        # 코드 생성
        result = generator.generate_code(
            instruction=args.instruction,
            code_type=args.code_type,
            max_tokens=args.max_tokens,
            temperature=args.temperature
        )
        
        # 파일로 저장
        filename = f"{args.code_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        saved_path = generator.save_code(result, args.code_type, filename, args.output_dir)
        
        print(f"\n생성된 코드:")
        print("-" * 70)
        print(result)
        print("-" * 70)
        
        # HTML인 경우 스크린샷 생성
        if args.screenshot and args.code_type == "HTML":
            generator.create_screenshot(saved_path, args.output_dir)



# 대화형 모드
def interactive_mode():
    """대화형 코드 생성 모드 - HTML/CSS/JS 파일 생성 및 저장"""
    print("=" * 70)
    print("CodeLlama 대화형 코드 생성기")
    print("=" * 70)
    
    model_path = input("모델 경로 (기본값: ./codellama_7b_finetuned): ").strip()
    if not model_path:
        model_path = "./codellama_7b_finetuned"
    
    output_dir = input("결과 저장 폴더 (기본값: ./result): ").strip()
    if not output_dir:
        output_dir = "./result"
    
    generator = CodeGenerator(model_path)
    
    print("\n사용 방법:")
    print("  - 프로젝트 이름 입력 시 HTML/CSS/JS 모두 생성")
    print("  - 또는 개별 코드 타입 선택: HTML, CSS, JS")
    print("  - 종료하려면 'exit' 입력\n")
    
    while True:
        print("-" * 70)
        
        # 프로젝트 전체 생성 또는 개별 생성 선택
        choice = input("1: 프로젝트 전체(HTML/CSS/JS), 2: 개별 파일 [1/2]: ").strip()
        
        if choice.lower() == "exit":
            break
        
        # 프로젝트 전체 생성
        if choice == "1":
            project_name = input("프로젝트 이름: ").strip()
            if project_name.lower() == "exit":
                break
            if not project_name:
                print("프로젝트 이름을 입력해주세요.")
                continue
            
            description = input("프로젝트 설명: ").strip()
            if description.lower() == "exit":
                break
            
            print("\n생성 중...\n")
            results = generator.generate_all_types(
                project_name=project_name,
                html_instruction=f"Create a complete HTML page for {description or project_name}",
                css_instruction=f"Create CSS styles for {description or project_name}",
                js_instruction=f"Create JavaScript functionality for {description or project_name}",
                output_dir=output_dir
            )
            
            # 스크린샷 생성 여부
            create_ss = input("\n스크린샷을 생성하시겠습니까? [y/N]: ").strip().lower()
            if create_ss == 'y' and results.get('html'):
                generator.create_screenshot(results['html']['path'], output_dir)
        
        # 개별 파일 생성
        else:
            code_type = input("코드 타입 (HTML/CSS/JS) [기본값: HTML]: ").strip().upper()
            if code_type == "EXIT":
                break
            if not code_type or code_type not in ["HTML", "CSS", "JS"]:
                code_type = "HTML"
            
            instruction = input("코드 설명: ").strip()
            if instruction.lower() == "exit":
                break
            if not instruction:
                print("설명을 입력해주세요.")
                continue
            
            print("\n생성 중...\n")
            result = generator.generate_code(
                instruction=instruction,
                code_type=code_type
            )
            
            # 파일 저장
            filename = f"{code_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            saved_path = generator.save_code(result, code_type, filename, output_dir)
            
            print("\n생성된 코드:")
            print("=" * 70)
            print(result)
            print("=" * 70 + "\n")
            
            # HTML인 경우 스크린샷 생성 여부 확인
            if code_type == "HTML":
                create_ss = input("스크린샷을 생성하시겠습니까? [y/N]: ").strip().lower()
                if create_ss == 'y':
                    generator.create_screenshot(saved_path, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 인자가 없으면 대화형 모드
        interactive_mode()
    else:
        # 인자가 있으면 CLI 모드
        main()
