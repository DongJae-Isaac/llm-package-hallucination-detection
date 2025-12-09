# LLM Package Hallucination Study  
### LLM 모델 별 패키지 환각 취약점 분석 및 공급망 공격 실증 연구

---

## 👥 팀원
- **김동재** (소프트웨어학부, 20213107)
- **김민경** (소프트웨어학부, 20221828)
- **김태욱** (전자정보공학부, 20192581)
- **안준엽** (소프트웨어학부, 20211794)

---

## 📌 프로젝트 소개

본 프로젝트는 Large Language Model(LLM)이 코드 생성 과정에서 실제 존재하지 않는 패키지를 추천하는  
**패키지 환각(Package Hallucination)** 문제를 분석하고, 환각 패키지를 악성 패키지로 활용하여  
**공급망 공격(Supply Chain Attack)**이 실제로 가능한지 검증하는 연구입니다.

연구 목표:

- 다양한 LLM 모델의 패키지 추천 정확도 및 환각 발생률 비교
- System Prompt 유형에 따른 환각률 변화 분석
- 프롬프트 카테고리별 환각 패턴 분석
- Self-Refinement / RAG 기반 환각 감소 실험
- 환각 패키지를 활용한 **RDD(Remote Dependency Download) 공급망 공격 실증**

---

## 📑 목차
- [테스트 모델](#테스트-모델)
- [실험 설계](#실험-설계)
- [주요 결과](#주요-결과)
- [설치 및 실행](#설치-및-실행)
- [프로젝트 구조](#프로젝트-구조)
- [연구 방법론](#연구-방법론)
- [결과 분석](#결과-분석)
- [프로젝트 시나리오](#프로젝트-시나리오)
- [참고 자료](#참고-자료)
- [문의](#문의)

---

## 🚀 테스트 모델

- **Marin 8B** — marin-community/marin-8b-instruct  
- **Qwen 2.5 7B Turbo** — Qwen/Qwen2.5-7B-Instruct-Turbo  
- **Gemma 3n E4B-it** — Google/gemma-3n-E4B-it  
- **Mistral 7B** — Mistralai/Mistral-7B-Instruct-v0.2  
- **CodeLlama 7B** — codellama/CodeLlama-7b-Instruct-hf  
- **GPT-OSS** — OpenAI 기반 오픈소스 모델  

---

## 🧪 실험 설계

### 📌 데이터셋 구성
- **총 질문 수:** 20,855  
- **카테고리:** 9개  
- **System Prompt:** 4종  
- **LLM 모델:** 6종  
- **전체 프롬프트 실행 수:** 83,540  

---

### 📌 System Prompt 유형

| 유형 | 설명 |
|------|------|
| **CODE_GEN** | 코드 생성 중심, 패키지 언급 최소화 |
| **PKG_DETERMINE** | 필요한 패키지 식별 유도 |
| **PKG_RECOMMEND** | 패키지 추천을 유도 (가장 높은 환각률) |
| **STDONLY** | 외부 패키지 금지, 표준 라이브러리만 사용 |

---

### 📌 프롬프트 카테고리

| Category | 설명 | 개수 |
|----------|------|------|
| Frontend | React/Vue/빌드 환경 | 4,997 |
| Error_Handling | 빌드 및 모듈 오류 | 3,583 |
| Backend | DB/ORM/서버 로직 | 2,871 |
| Data_Processing | 파싱/크롤링 | 2,629 |
| Uncategorized | 기타 | 2,418 |
| Web_Development | HTTPS/웹크롤링 | 1,237 |
| Monitoring | 로그/이상 탐지 | 1,196 |
| App_Development | React Native/Electron | 833 |
| Prompt_Security | TLS/SSL, 취약점 스캔 | 751 |

---

## 📊 주요 결과

### 📌 모델별 환각 비율

| 모델 | 실제 패키지 | 환각 패키지 | 전체 | 환각 비율 |
|------|-------------|-------------|--------|-----------|
| **GPT-OSS** | 14,372 | 416 | 14,788 | **2.81%** |
| **Qwen** | 16,032 | 938 | 16,970 | **5.53%** |
| **Gemma** | 12,436 | 1,089 | 13,525 | **8.05%** |
| **CodeLlama** | 12,915 | 1,284 | 14,199 | **9.04%** |
| **Mistral** | 40,003 | 4,981 | 44,984 | **11.07%** |
| **Marin** | 13,428 | 3,097 | 16,525 | **18.74%** |

---

### 📌 핵심 발견사항

상세한 분석 그래프는 [실험 결과 문서](data/docs/experiment_results.md)를 참조하세요:

#### 1. 환각이 많이 발생한 카테고리
- **Monitoring**
- **Backend**
- **Web_Development**

#### 2. 환각이 적게 발생한 카테고리
- **Frontend**
- **Error_Handling**
- **Performance**

#### 3. System Prompt 영향
- **PKG_RECOMMEND**, **PKG_DETERMINE** → 환각률 가장 높음  
- **STDONLY** → 환각 최소화  
- **CODE_GEN** → 불필요한 패키지 언급 감소  

#### 4. 네이밍 유효성 및 Slopsquatting 위험
- 환각 패키지 **5,713개 중 98% 이상이 npm 네이밍 규칙 통과**
- 그 중 **73%가 실제 등록 가능 패키지**
- → 공격자가 즉시 악성 패키지로 등록할 수 있는 수준

#### 5. Socket.dev 기반 위험 패키지 검출
- 악성코드 의심 패키지 다수 발견
- 고위험 Typosquatting 패키지도 여러 모델에서 등장

---

## 🛠 Self-Refinement / RAG 결과

### 📌 Self-Refinement 적용 효과

| Model Name                    | Before        | After         | Reduction        |
|------------------------------|--------------|---------------|------------------|
| **gpt-oss-7B**               | 218 (0.26%)  | 92 (0.11%)    | 126 (-0.15%p)    |
| **gemma-3n-E4B-it**          | 661 (0.79%)  | 594 (0.71%)   | 57 (-0.08%p)     |
| **qwen2.5-7b-instruct-turbo**| 660 (0.79%)  | 279 (0.33%)   | 381 (-0.46%p)    |
| **marin-8b-instruct**        | 1,129 (1.35%)| 525 (0.63%)   | 604 (-0.72%p)    |
| **codellama-7B**             | 958 (1.15%)  | 948 (1.13%)   | 10 (-0.01%p)     |
| **mistral-7b-instruct-v0.2** | 3,518 (4.21%)| 2,638 (3.16%) | 880 (-1.05%p)    |

→ 모델이 응답을 재검토하는 과정에서 환각이 안정적으로 감소.

---

### 📌 RAG 적용 효과

| Model Name                    | Before        | After         | Reduction        |
|------------------------------|--------------|---------------|------------------|
| **gpt-oss-7B**               | 218 (0.26%)  | 33 (0.04%)    | 185 (-0.22%p)    |
| **gemma-3n-E4B-it**          | 661 (0.79%)  | 99 (0.12%)    | 562 (-0.67%p)    |
| **qwen2.5-7b-instruct-turbo**| 660 (0.79%)  | 119 (0.14%)   | 541 (-0.65%p)    |
| **marin-8b-instruct**        | 1,129 (1.35%)| 203 (0.24%)   | 926 (-1.11%p)    |
| **codellama-7B**             | 958 (1.15%)  | 172 (0.21%)   | 786 (-0.94%p)    |
| **mistral-7b-instruct-v0.2** | 3,518 (4.21%)| 704 (0.84%)   | 2,814 (-3.37%p)  |

→ 가장 강력한 환각 억제 효과.  
→ 패키지 추천 기능에는 RAG 기반 실시간 검증이 필수적임을 시사.

---

## 🔥 RDD 공급망 공격 실증

환각 패키지를 실제 악성 패키지로 제작하여 공격을 수행한 결과:

### 📌 공격 1: 환경 변수 탈취
패키지 설치만으로 다음 정보 탈취 가능:

- OS/Node 환경 정보  
- 환경 변수(API 키 포함)  
- npm 설정  
- 시스템 경로  

### 📌 공격 2: Reverse Shell
악성 패키지 설치 → 공격자에게 자동 연결

공격자가 수행 가능한 작업:

- 내부망 스캔  
- 권한 상승  
- 중요 파일 탈취  
- 지속적인 백도어 생성  

→ 패키지 환각이 실제 침해사고로 이어지는 공격 벡터임을 실증.

---

## ⚙️ 설치 및 실행

### 📌 요구사항
```bash
python --version   # Python 3.8 이상
node --version
```
### 📌 설치 방법

```bash
git clone https://github.com/DongJae-Isaac/llm-package-hallucination-detection.git
cd llm-package-hallucination-detection
pip install pandas numpy matplotlib seaborn requests

# 데이터 파일 다운로드 후 압축 해제
unzip data_files.zip
```

### 📌 실행 방법

``` bash
1. 패키지 추출 및 검증
python src/detection/prompt_detection.py
python reference_code/package_detection.py

2. 결과 분석
python src/analysis/test5.py
python src/analysis/category_analysis.py

3. LLM 테스트(선택)
python src/llm_test/ollama_codellama_test.py
python src/llm_test/together_ai_test_m.py
```

### 📁 프로젝트 구조

```plaintxt
llm-package-hallucination-detection/
├── data/
│   ├── docs/
│   ├── prompts/
│   └── reference/
├── results/
├── src/
│   ├── analysis/
│   ├── detection/
│   └── llm_test/
├── reference_code/
└── requirements.txt
```

## 🧬 연구 방법론 요약

### 데이터 수집

- 20,855개 질문 생성
- 9개 카테고리 분류
- 4종 System Prompt 조합

### 패키지 추출

``` javascript
npm install <package-name>
require('<package-name>')
import pkg from '<package-name>'
```

### 검증 파이프라인

- 패턴 기반 패키지명 추출
- Node 내장 모듈 필터링
- npm Registry 조회
- 환각 판단 및 통계화

### 분석 지표
- 환각 비율
- 카테고리별 환각 패턴
- System Prompt 영향
- 모델 간 성능 비교

## 📈 결과 분석 요약
- Monitoring/Backend 분야에서 가장 높은 환각률
- 패키지 추천·의존성 판단 프롬프트가 가장 위험
- 환각 패키지 다수가 npm에 즉시 등록 가능
- Self-Refinement 및 RAG는 환각 감소에 효과적
- 환각 패키지를 통한 RDD 공격이 실제 성공

## 🔗 프로젝트 시나리오

환경변수 유출 및 Reverse Shell 공격 PoC
https://github.com/JunYeopAn/llm-package-hallucination-attack-scenario.git

## 📚 참고 자료

https://github.com/Spracks/PackageHallucination
https://zenodo.org/records/14676377
