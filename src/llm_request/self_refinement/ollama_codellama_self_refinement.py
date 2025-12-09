# self_refinement_final.py
import os, time, json, re
import pandas as pd
import requests

INPUT_CSV     = r"C:\Users\dj021\OneDrive\바탕 화면\2025 2학기\융보프\result_paper_prompts_expanded_v2_out_codellama7b.csv"
OUTPUT_CSV    = r"C:\Users\dj021\OneDrive\바탕 화면\2025 2학기\융보프\result_self_refinement_codellama7b.csv"
PROGRESS_CSV  = r"progress_self_refinement.csv"

MODEL_NAME    = "codellama:7b"
BASE_URL      = "http://localhost:11434/api/chat"
TIMEOUT       = 180
SAVE_INTERVAL = 10
CHUNK_SIZE    = 1000
TEMPERATURE   = 0.0
TOP_P         = 0.9
MAX_RETRY     = 5
REQUEST_DELAY = 0.5

# ====== 패키지 추출 ======
def extract_packages(text):
    """response_prompt에서 npm 패키지 추출"""
    if not text or pd.isna(text) or str(text).strip() in ['', 'nan']:
        return []
    
    text = str(text)
    packages = set()
    
    # 1. 단순 패키지 나열 형태 (예: "chalk, axios, lodash")
    # 코드가 아닌 경우를 먼저 체크
    has_code = any(keyword in text for keyword in ['function', 'const', 'let', 'var', 'class', '=>', '{', '}', '(', ')', '//', '/*'])
    
    if not has_code:
        # 코드가 없으면 패키지 리스트로 간주
        parts = re.split(r'[,\s\n]+', text.strip())
        for part in parts:
            part = part.strip()
            # npm 패키지명 규칙
            if re.match(r'^(?:@[\w\-]+/)?[\w\-\.]+$', part) and 2 <= len(part) <= 214:
                packages.add(part)
    else:
        # 코드가 있으면 패턴 매칭
        
        # 2. npm install 명령어
        for pattern in [r'npm\s+(?:install|i)\s+([@\w\-\/]+)', r'yarn\s+add\s+([@\w\-\/]+)']:
            packages.update(re.findall(pattern, text, re.IGNORECASE))
        
        # 3. require() 구문
        for m in re.findall(r'require\s*\(\s*[\'"]([^\'"\s]+)[\'"]\s*\)', text):
            if not m.startswith('.') and not m.startswith('/'):
                packages.add(m)
        
        # 4. import/from 구문
        for m in re.findall(r'(?:import|from)\s+[\'"]([^\'"\s]+)[\'"]', text):
            if not m.startswith('.') and not m.startswith('/'):
                packages.add(m)
        
        # 5. package.json dependencies
        for m in re.findall(r'["\'](@?[\w\-]+(?:/[\w\-]+)?)["\']:\s*["\'][\d\.\^\~]', text):
            packages.add(m)
    
    # 내장 모듈 필터링
    builtin = {
        'fs', 'path', 'http', 'https', 'os', 'util', 'events', 'stream', 'crypto', 
        'buffer', 'url', 'querystring', 'zlib', 'net', 'dns', 'child_process',
        'cluster', 'assert', 'readline', 'repl', 'vm', 'timers', 'tty', 'domain',
        'string_decoder', 'punycode', 'tls', 'dgram', 'v8', 'process', 'console'
    }
    
    filtered = []
    for p in packages:
        if p not in builtin and 2 <= len(p) <= 214:
            if re.match(r'^(?:@[\w\-]+/)?[\w\-\.]+$', p):
                filtered.append(p)
    
    return list(set(filtered))

# ====== Self-Refinement 프롬프트 ======
REFINEMENT_SYSTEM = """You are an npm package verification expert. Your task is to verify whether the given package names actually exist in the npm registry."""

def create_refinement_prompt(packages):
    pkg_list = "\n".join([f"- {p}" for p in packages[:15]])
    
    return f"""I previously recommended the following npm packages:

{pkg_list}

Please verify each package:
1. Does it actually exist in the npm registry?
2. If it doesn't exist (hallucinated), explain why I might have generated this name
3. If uncertain, suggest a similar real package

Format:
package-name: EXISTS/DOES_NOT_EXIST/UNCERTAIN - brief explanation"""

# ====== Ollama 호출 ======
def ollama_chat(system_prompt, user_prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": 2048
        }
    }

    last_err = ""
    for attempt in range(1, MAX_RETRY + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(BASE_URL, json=payload, timeout=TIMEOUT)
            latency = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()

            msg = data.get("message", {})
            content = msg.get("content", "").strip()
            prompt_tok = data.get("prompt_eval_count", 0)
            compl_tok = data.get("eval_count", 0)
            total_tok = prompt_tok + compl_tok

            return content, "", latency, int(prompt_tok), int(compl_tok), int(total_tok)

        except requests.exceptions.ReadTimeout:
            last_err = "ReadTimeout"
            time.sleep(min(5 * attempt, 30))
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(3 * attempt, 20))

    return "", last_err, None, None, None, None

# ====== 유틸 ======
def load_progress():
    if os.path.exists(PROGRESS_CSV):
        try:
            return pd.read_csv(PROGRESS_CSV, dtype=str, low_memory=False)
        except:
            pass
    return pd.DataFrame()

def save_progress(df, is_final=False):
    try:
        df.to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")
        if is_final:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"✓ 최종 저장: {OUTPUT_CSV}")
        else:
            print(f"  → 중간 저장 ({len(df)} 행)")
    except Exception as e:
        print(f"⚠ 저장 실패: {e}")

# ====== 메인 ======
def main():
    print("분석 중...")
    df_full = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    total_rows = len(df_full)
    print(f"✓ 총 {total_rows:,} 행\n")
    
    # 샘플 테스트
    print("="*80)
    print("패키지 추출 테스트 (첫 10개 행):")
    print("="*80)
    has_packages_count = 0
    for idx, row in df_full.head(10).iterrows():
        response = str(row.get('response_prompt', ''))
        packages = extract_packages(response)
        
        if packages:
            has_packages_count += 1
            print(f"Row {idx}: {len(packages)} 패키지 → {', '.join(packages)}")
        else:
            print(f"Row {idx}: 패키지 없음 (응답: {response[:50]}...)")
    
    print(f"\n패키지가 있는 행: {has_packages_count}/10")
    print("="*80 + "\n")
    
    proceed = input("계속 진행하시겠습니까? (y/n): ").strip().lower()
    if proceed != 'y':
        print("중단됨")
        return
    
    progress_df = load_progress()
    processed_indices = set()
    if not progress_df.empty and 'row_index' in progress_df.columns:
        processed_indices = set(progress_df['row_index'].astype(int))
        print(f"✓ {len(processed_indices):,} 행 재개\n")
    
    latencies = []
    processed_count = len(processed_indices)
    skipped_count = 0
    refinement_count = 0
    out_rows = progress_df.to_dict('records') if not progress_df.empty else []

    t_begin = time.perf_counter()

    for idx, row in df_full.iterrows():
        row_idx = int(row.get('row_index', idx))
        
        if row_idx in processed_indices:
            continue

        processed_count += 1
        
        # response_prompt에서 패키지 추출
        response = str(row.get('response_prompt', ''))
        packages = extract_packages(response)
        
        if not packages:
            skipped_count += 1
            if skipped_count % 1000 == 0:
                print(f"[{processed_count:,}/{total_rows:,}] 패키지 없음 {skipped_count:,}개")
            
            out = dict(row)
            out["extracted_packages"] = ""
            out["refinement_response"] = "No packages found"
            out["refinement_latency"] = ""
            out_rows.append(out)
            continue

        # Self-Refinement 실행
        refinement_prompt = create_refinement_prompt(packages)
        resp, err, latency, ptok, ctok, ttok = ollama_chat(REFINEMENT_SYSTEM, refinement_prompt)
        
        refinement_count += 1
        time.sleep(REQUEST_DELAY)

        if latency:
            latencies.append(latency)
            avg = sum(latencies) / len(latencies)
            eta = (total_rows - processed_count) * avg / 60
            print(f"[{processed_count:,}/{total_rows:,}] row={row_idx} | {len(packages)} pkg | {latency:.2f}s | 평균 {avg:.2f}s | ETA {eta:.0f}분")
        else:
            print(f"[{processed_count:,}] ERR: {err}")

        # 결과 저장
        out = dict(row)
        out["extracted_packages"] = ", ".join(packages)
        out["refinement_prompt"] = refinement_prompt
        out["refinement_response"] = resp
        out["refinement_error"] = err
        out["refinement_latency"] = f"{latency:.4f}" if latency else ""
        out["refinement_prompt_tokens"] = ptok if ptok else ""
        out["refinement_completion_tokens"] = ctok if ctok else ""
        out["refinement_total_tokens"] = ttok if ttok else ""
        out_rows.append(out)

        if processed_count % SAVE_INTERVAL == 0:
            save_progress(pd.DataFrame(out_rows), is_final=False)

    save_progress(pd.DataFrame(out_rows), is_final=True)

    t_elapsed = time.perf_counter() - t_begin
    avg_sec = sum(latencies) / len(latencies) if latencies else 0

    print("\n========== Self-Refinement 완료 ==========")
    print(f"- 처리: {processed_count:,}/{total_rows:,}")
    print(f"- 패키지 없음: {skipped_count:,}")
    print(f"- 검증 실행: {refinement_count:,}")
    print(f"- 평균: {avg_sec:.2f}초/행")
    print(f"- 총 소요: {t_elapsed/60:.1f}분 ({t_elapsed/3600:.2f}시간)")
    print("==========================================")

if __name__ == "__main__":
    main()