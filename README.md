# LLM Package Hallucination Study

## 프로젝트 소개
LLM이 npm 패키지를 추천할 때 발생하는 환각(hallucination) 현상을 연구하는 프로젝트입니다.
[참고 논문](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-742-spracklen.pdf)을 기반으로 진행되었습니다.

### 팀원
- 김동재 (소프트웨어학부, 20213107)
- 김민경 (소프트웨어학부, 20221828)
- 김태욱 (전자정보공학부, 20192581)
- 안준엽 (소프트웨어학부, 20211794)

## 연구 목적
- 다양한 LLM 모델의 패키지 추천 정확도 측정
- System prompt에 따른 환각 발생률 비교
- 프롬프트 카테고리별 환각 패턴 분석
- Self-Refinement 기법을 통한 환각 감소 방안 연구

## 목차
- [테스트 모델](#테스트-모델)
- [실험 설계](#실험-설계)
- [주요 결과](#주요-결과)
- [설치 및 실행](#설치-및-실행)
- [프로젝트 구조](#프로젝트-구조)
- [연구 방법론](#연구-방법론)
- [결과 분석](#결과-분석)
- [참고 자료](#참고-자료)

## 테스트 모델
- **Marin** - marin-community/marin-8b-instruct
- **Qwen** - Qwen/Qwen2.5-7B-Instruct-Turbo
- **Gemma** - Google/gemma-3n-E4B-it
- **Mistral** - Mistralai/Mistral-7B-Instruct-v0.2
- **CodeLlama 7B** - codellama/CodeLlama-7b-Instruct-hf
- **GPT-OSS** - (OpenAI 기반 오픈소스 모델)

## 실험 설계

### 데이터셋 규모
- **총 질문 수**: 20,855개
- **카테고리**: 9개
- **System Prompt 유형**: 4가지
- **테스트 모델**: 6개

### System Prompt 유형
1. **Default**: 기본 프롬프트
2. **Security-focused**: 보안 중심 프롬프트
3. **Best-practices**: 모범 사례 중심
4. **Minimal**: 최소한의 지시사항

### 프롬프트 카테고리
| Category | 설명 | 예시 개수 |
|----------|------|-----------|
| Frontend | React, Vue, 빌드 도구 등 | 4,997 |
| Error_Handling | 빌드 실패, 모듈 오류 등 | 3,583 |
| Backend | DB, ORM, 서버 로직 등 | 2,871 |
| Data_Processing | 파싱, 크롤링 등 | 2,629 |
| Uncategorized | 기타 | 2,418 |
| Web_Development | 웹크롤링, HTTPS, 웹 접근성 | 1,237 |
| Monitoring | 실시간 이상 탐지, 로깅 | 1,196 |
| App_Development | 모바일/데스크톱 앱 개발 | 833 |
| Prompt_Security | TLS/SSL, 취약점 스캔 | 751 |
| Performance | 트래픽 관리, 최적화 | 370 |

## 주요 결과

### 모델별 환각 비율
| 모델 | 실제 패키지 | 환각 패키지 | 전체 패키지 | 환각 비율 |
|------|-------------|-------------|-------------|-----------|
| **GPT-OSS** | 14,372 | 416 | 14,788 | **2.81%** |
| **Qwen** | 16,032 | 938 | 16,970 | **5.53%** |
| **Gemma** | 12,436 | 1,089 | 13,525 | **8.05%** |
| **CodeLlama 7B** | 12,915 | 1,284 | 14,199 | **9.04%** |
| **Mistral** | 40,003 | 4,981 | 44,984 | **11.07%** |
| **Marin** | 13,428 | 3,097 | 16,525 | **18.74%** |

### 주요 발견사항
1. **GPT-OSS**가 가장 낮은 환각 비율(2.81%)을 보임
2. **Marin** 모델이 가장 높은 환각 비율(18.74%)을 기록
3. 카테고리별로 환각 발생 패턴이 상이함
4. System prompt 유형에 따라 환각 비율 변화 확인

### 시각화 결과
상세한 분석 그래프는 [실험 결과 문서](data/docs/experiment_results.md)를 참조하세요:
- 모델별 카테고리 환각 비율
- 시스템 프롬프트별 환각 비율
- 카테고리 × 프롬프트 조합 히트맵

## 설치 및 실행

### 필수 요구사항
```bash
# Python 3.8 이상
python --version

# Node.js (패키지 검증용)
node --version
```

### 설치 방법
```bash
# 1. 저장소 클론
git clone https://github.com/DongJae-Isaac/llm-package-hallucination-detection.git
cd llm-package-hallucination-detection

# 2. Python 의존성 설치
pip install pandas numpy matplotlib seaborn requests

# 3. Releases에서 data_files.zip 다운로드
# https://github.com/DongJae-Isaac/llm-package-hallucination-detection/releases/latest

# 4. 압축 해제
unzip data_files.zip
```

### 실행 방법

#### 1. 패키지 추출 및 검증
```bash
# LLM 응답에서 패키지 추출
python src/detection/prompt_detection.py

# npm 레지스트리를 통한 검증
python reference_code/package_detection.py
```

#### 2. 결과 분석
```bash
# 통계 분석 및 시각화
python src/analysis/test5.py

# 카테고리별 분석
python src/analysis/category_analysis.py
```

#### 3. LLM 테스트 (선택사항)
```bash
# Ollama 로컬 테스트
python src/llm_test/ollama_codellama_test.py

# Together AI API 테스트
python src/llm_test/together_ai_test_m.py
```

## 프로젝트 구조
```
llm-package-hallucination-detection/
├── data/
│   ├── docs/                    # 문서
│   │   ├── experiment_results.md
│   │   └── project_timeline.md
│   ├── prompts/                 # 테스트 프롬프트
│   │   ├── paper-prompts.csv
│   │   └── paper_prompts_expanded_v2.csv
│   └── reference/               # 논문 데이터
│       └── npm_package_names.csv
├── results/
│   ├── analysis/                # 최종 분석 결과
│   │   ├── FINAL_verified_libraries_v7.csv
│   │   └── FINAL_verified_npm_by_system.csv
│   ├── gemma/                   # Gemma 모델 결과
│   ├── gpt_oss/                 # GPT-OSS 모델 결과
│   ├── marin/                   # Marin 모델 결과
│   ├── mistral/                 # Mistral 모델 결과
│   ├── ollama/                  # CodeLlama 7B 결과
│   └── qwen/                    # Qwen 모델 결과
├── src/
│   ├── analysis/                # 결과 분석 코드
│   │   └── test5.py
│   ├── detection/               # 패키지 추출 및 검증
│   │   └── prompt_detection.py
│   └── llm_test/                # LLM 테스트 코드
│       ├── ollama_codellama_test.py
│       └── together_ai_test_m.py
├── reference_code/              # 논문 코드 구현
│   ├── generate_package_names.py
│   ├── package_detection.py
│   └── run_test.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 연구 방법론

### 1. 데이터 수집
- 원본 논문의 데이터셋을 기반으로 20,855개 질문 구성
- 9개 카테고리로 분류
- 4가지 system prompt 유형 적용

### 2. 패키지 추출
LLM 응답에서 다음 패턴을 통해 패키지 추출:
```python
# npm install 명령어
npm install <package-name>

# require 구문
const pkg = require('<package-name>')

# import 구문  
import pkg from '<package-name>'
```

### 3. 검증 파이프라인 (5단계)
1. **패턴 매칭**: 응답에서 패키지명 후보 추출
2. **내장 모듈 필터링**: Node.js 기본 모듈 제외
3. **npm Registry 조회**: 실제 존재 여부 확인
4. **환각 판정**: 존재하지 않는 패키지 식별
5. **통계 분석**: 모델/카테고리별 집계

### 4. 분석 지표
- **환각 비율**: (환각 패키지 수) / (전체 패키지 수)
- **카테고리별 환각 패턴**
- **System prompt 영향도**
- **모델 간 비교 분석**

## 결과 분석

### 핵심 발견사항

1. **모델 성능 차이**
   - 상용 모델(GPT-OSS)이 오픈소스 모델 대비 낮은 환각 비율
   - 모델 크기가 반드시 성능과 비례하지 않음

2. **카테고리 영향**
   - Frontend, Backend 카테고리에서 상대적으로 높은 정확도
   - Error_Handling, Uncategorized에서 환각 빈도 증가

3. **System Prompt 효과**
   - Security-focused prompt가 일부 모델에서 환각 감소 효과
   - 프롬프트 유형에 따른 모델별 반응 차이 존재

4. **Self-Refinement 가능성**
   - 초기 실험 결과 환각 감소 효과 확인
   - 추가 연구 진행 중

## 프로젝트 시나리오  
- [환경변수 유출 RDD 공격] https://github.com/JunYeopAn/llm-package-hallucination-attack-scenario.git

## 라이선스

이 프로젝트는 학술 연구 목적으로 제작되었습니다.

## 참고 자료
- [원본 논문](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-742-spracklen.pdf)
- [논문 GitHub](https://github.com/Spracks/PackageHallucination)
- [데이터셋 출처](https://zenodo.org/records/14676377)
- [프로젝트 타임라인](data/docs/project_timeline.md)

## 문의
프로젝트 관련 문의는 [Issues](https://github.com/DongJae-Isaac/llm-package-hallucination-detection/issues)를 통해 남겨주세요.
