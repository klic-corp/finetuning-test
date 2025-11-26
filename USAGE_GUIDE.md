# 코드 파인튜닝 및 자동 생성 사용 가이드

## 기능

- **HTML, CSS, JS 코드 자동 생성**
- **HTML 스크린샷 자동 생성**
- **CLI 모드와 대화형 모드 지원**

## 설치

### 필수 라이브러리
```powershell
pip install torch transformers peft accelerate bitsandbytes safetensors
```

### 스크린샷 생성을 위한 추가 라이브러리 (선택사항)
```powershell
pip install selenium webdriver-manager
```

## 사용 방법

### 1. 대화형 모드 (권장)

```powershell
python generate_code.py
```

#### 전체 프로젝트 생성 (HTML + CSS + JS)
```
1: 프로젝트 전체(HTML/CSS/JS), 2: 개별 파일 [1/2]: 1
프로젝트 이름: my_website
프로젝트 설명: responsive landing page with navigation
스크린샷을 생성하시겠습니까? [y/N]: y
```

결과:
```
result/
├── my_website.html
├── my_website.css
├── my_website.js
└── my_website_screenshot.png
```

#### 개별 파일 생성
```
1: 프로젝트 전체(HTML/CSS/JS), 2: 개별 파일 [1/2]: 2
코드 타입 (HTML/CSS/JS) [기본값: HTML]: HTML
코드 설명: contact form with validation
스크린샷을 생성하시겠습니까? [y/N]: y
```

결과:
```
result/
├── html_20251106_143022.html
└── html_20251106_143022_screenshot.png
```

---

### 2. CLI 모드

#### 전체 프로젝트 생성 (HTML + CSS + JS)
```powershell
python generate_code.py --project_name "portfolio" --instruction "전문적으로 보이는 포트폴리오 사이트" --screenshot
```

또는

```powershell
python generate_code.py --code_type ALL --instruction "e-커머스 제품 페이지" --screenshot
```

#### 개별 파일 생성

##### HTML 생성
```powershell
python generate_code.py --code_type HTML --instruction "반응형 드롭다운 메뉴바" --screenshot
```

##### CSS 생성
```powershell
python generate_code.py --code_type CSS --instruction "모던하고 어두운 테마의 스타일"
```

##### JavaScript 생성
```powershell
python generate_code.py --code_type JS --instruction "슬라이드 기능이 있는 배너"
```

---

### 3. CLI 옵션 설명

```powershell
python generate_code.py --help
```

주요 옵션:

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model_path` | 파인튜닝된 모델 경로 | `./codellama_7b_finetuned` |
| `--project_name` | 프로젝트 이름 (HTML/CSS/JS 모두 생성) | None |
| `--code_type` | 코드 타입 (HTML/CSS/JS/ALL) | `HTML` |
| `--instruction` | 생성할 코드 설명 | - |
| `--output_dir` | 결과 저장 디렉토리 | `./result` |
| `--screenshot` | 스크린샷 생성 | `False` |
| `--max_tokens` | 최대 생성 토큰 수 | `1024` |
| `--temperature` | 생성 다양성 (0.0-1.0) | `0.7` |

---

## 예제

### 예제 1: 랜딩 페이지 전체 생성
```powershell
python generate_code.py --project_name "landing_page" --instruction "초등학교 테마의 현대적인 랜딩 페이지" --screenshot
```

생성 파일:
- `result/landing_page.html`
- `result/landing_page.css`
- `result/landing_page.js`
- `result/landing_page_screenshot.png`

### 예제 2: 로그인 폼 HTML만 생성
```powershell
python generate_code.py --code_type HTML --instruction "이메일과 비밀번호 폼이 있는 로그인 페이지" --screenshot --output_dir "./my_results"
```

### 예제 3: 애니메이션 CSS만 생성
```powershell
python generate_code.py --code_type CSS --instruction "부드럽게 페이드인 되는 카드박스"
```

### 예제 4: 슬라이더 JS만 생성
```powershell
python generate_code.py --code_type JS --instruction "자동 플레이 기능이 있는 이미지 슬라이더"
```

---

## 생성된 파일 구조

### 프로젝트 전체 생성 시
```
result/
├── project_name.html
├── project_name.css
├── project_name.js
└── project_name_screenshot.png
```

### 개별 파일 생성 시
```
result/
├── html_20251106_143022.html
├── html_20251106_143022_screenshot.png
├── css_20251106_143523.css
└── js_20251106_144012.js
```

---

## 스크린샷 생성

### 필요 조건
```powershell
pip install selenium webdriver-manager
```

### 자동 작동
- Chrome 브라우저 필요
- ChromeDriver 자동 다운로드 및 설치
- 해상도: 1920x1080
- 헤드리스 모드로 실행 (백그라운드)

### 스크린샷 생성 안 될 때
1. Chrome 브라우저 설치 확인
2. 라이브러리 재설치:
   ```powershell
   pip install --upgrade selenium webdriver-manager
   ```
3. 수동으로 ChromeDriver 다운로드 필요 시

---

## 출력 디렉토리 변경

기본 `result` 폴더 대신 다른 폴더 사용:

### CLI 모드
```powershell
python generate_code.py --output_dir "./my_outputs" --project_name "website"
```

### 대화형 모드
```
결과 저장 폴더 (기본값: ./result): ./my_outputs
```

---

## 고급 설정

### 생성 품질 조정

#### 더 창의적인 코드 (temperature 높임)
```powershell
python generate_code.py --temperature 0.9 --instruction "독창적인 내비게이션 디자인"
```

#### 더 보수적인 코드 (temperature 낮춤)
```powershell
python generate_code.py --temperature 0.3 --instruction "일반적인 문의 폼"
```

### 더 긴 코드 생성
```powershell
python generate_code.py --max_tokens 2048 --instruction "완성형 대시보드 레이아웃"
```

---

## 문제 해결

### 메모리 부족 오류
- `max_tokens` 값 줄이기: `--max_tokens 512`
- 모델 로드 시 자동으로 4-bit 양자화 적용됨

### 스크린샷 생성 실패
- Chrome 브라우저 설치 확인
- selenium 재설치: `pip install --upgrade selenium webdriver-manager`

### 생성된 코드 품질 개선
- `temperature` 조정 (0.5-0.8 권장)
- 더 구체적인 instruction 작성
- 예: ❌ "navbar" → ✅ "드롭다운 메뉴와 모바일 햄버거가 있는 반응형 탐색 바"

---

## 팁

1. **구체적인 설명 작성**
   - 나쁨: "button"
   - 좋음: "modern gradient button with hover animation"

2. **프로젝트 전체 생성 활용**
   - HTML, CSS, JS를 각각 생성하는 것보다 한 번에 생성하면 일관성 있음

3. **스크린샷으로 확인**
   - HTML 생성 후 바로 시각적으로 확인 가능

4. **결과 폴더 정리**
   - 프로젝트별로 output_dir 분리 권장