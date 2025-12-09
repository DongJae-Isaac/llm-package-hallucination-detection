# Self-Refinement for Hallucinated NPM Packages
# Based on together_ai_test_m.py structure
# Filters and processes only rows with hallucinated packages
# 
# Usage: python together_ai_self_refinement_v2.py <model_name> <together_model> <api_key_num>
# Example: python together_ai_self_refinement_v2.py gemma "google/gemma-3n-E4B-it" 1

import os, time, sys, json, math, re, ast
import pandas as pd
import requests
from pathlib import Path

# ============= API Keys =============
API_KEYS = {
    "1": "",
    "2": "",
    "3": "",
    "4": "",
}

# ============= Parse Command Line Arguments =============
if len(sys.argv) != 4:
    print("Usage: python together_ai_self_refinement_v2.py <model_name> <together_model> <api_key_num>")
    print("Example: python together_ai_self_refinement_v2.py gemma \"google/gemma-3n-E4B-it\" 1")
    print("\nAvailable model_name options: gemma, gpt_oss, marin, mistral, qwen")
    print("Available api_key_num: 1, 2, 3, 4")
    sys.exit(1)

MODEL_NAME = sys.argv[1]
TOGETHER_MODEL = sys.argv[2]
API_KEY_NUM = sys.argv[3]

if API_KEY_NUM not in API_KEYS:
    print(f"❌ Invalid API key number: {API_KEY_NUM}. Must be 1, 2, 3, or 4.")
    sys.exit(1)

API_KEY = API_KEYS[API_KEY_NUM]

# ============= Configuration =============
INPUT_CSV   = rf"d:\slopsquatting\result\result_paper_prompts_expanded_v2_out_{MODEL_NAME}_final.csv"
OUTPUT_CSV  = rf"d:\slopsquatting\result\self-refinement\result_self_refinement_{MODEL_NAME}.csv"
PROGRESS_CSV = rf"d:\slopsquatting\progress_self_refinement_{MODEL_NAME}.csv"

CHUNK_SIZE = 100
BASE_URL    = "https://api.together.xyz/v1/chat/completions"

# gpt_oss는 더 긴 타임아웃 설정
if MODEL_NAME == "gpt_oss":
    TIMEOUT = 600  # 10분
    MAX_RETRIES = 5  # 재시도 횟수
    RETRY_DELAY = 10  # 재시도 대기 시간 (초)
    print(f"⚙️  gpt_oss 모드: 타임아웃 {TIMEOUT}초, 최대 재시도 {MAX_RETRIES}회")
else:
    TIMEOUT = 300
    MAX_RETRIES = 3
    RETRY_DELAY = 5

SAVE_INTERVAL = 10

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Model context limits
MODEL_MAX_CONTEXT = 4096 
DEFAULT_MAX_COMPLETION = 2048
TOKEN_BUFFER = 100

# ============= Token Counting =============
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")  
    def approx_tokens(text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        return len(enc.encode(text))
except Exception:
    def approx_tokens(text: str) -> int:
        if not isinstance(text, str) or not text:
            return 0
        return int(len(text.split()) * 1.3)

# ============= Hallucination Detection =============
def has_hallucinated_packages_from_csv(row):
    """
    CSV의 response_npm_hallucinated 컬럼에서 환각 패키지 확인
    Returns: (has_hallucination: bool, packages: list)
    """
    try:
        hallucinated_str = row.get('response_npm_hallucinated', '')
        
        if pd.isna(hallucinated_str) or hallucinated_str == '' or hallucinated_str == '[]':
            return False, []
        
        packages = ast.literal_eval(hallucinated_str)
        
        if isinstance(packages, list) and len(packages) > 0:
            return True, packages
        
        return False, []
    except Exception as e:
        return False, []

# ============= Refinement Prompt Creation =============
def create_refinement_prompt(packages):
    """Self-refinement 프롬프트 생성"""
    pkg_list = "\n".join([f"- {p}" for p in packages[:20]])  # 최대 20개
    
    return f"""I previously recommended the following npm packages:

{pkg_list}

Please verify each package:
1. Does it actually exist in the npm registry?
2. If it doesn't exist (hallucinated), explain why I might have generated this name
3. If uncertain, suggest a similar real package

Format:
package-name: EXISTS/DOES_NOT_EXIST/UNCERTAIN - brief explanation"""

# ============= Together AI API Call =============
def together_chat(system_prompt: str, user_prompt: str):
    """Together chat.completions 호출 + usage/latency 반환"""
    
    # 1. 로컬의 근사치로 첫 시도 값을 계산
    approx_input = f"[SYSTEM]\n{system_prompt or ''}\n[/SYSTEM]\n[USER]\n{user_prompt or ''}\n[/USER]"
    prompt_tokens_approx = approx_tokens(approx_input)
    available_for_completion = MODEL_MAX_CONTEXT - prompt_tokens_approx - TOKEN_BUFFER

    if available_for_completion <= 0:
        err_msg = f"Pre-flight Error: Approx prompt tokens ({prompt_tokens_approx}) exceed model limit ({MODEL_MAX_CONTEXT})."
        print(f"    -> {err_msg}")
        return "", err_msg, 0, prompt_tokens_approx, 0, prompt_tokens_approx

    final_max_tokens = min(DEFAULT_MAX_COMPLETION, available_for_completion)
    
    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user",   "content": user_prompt or ""},
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": final_max_tokens
    }
    
    last_err = ""
    current_prompt_tokens = prompt_tokens_approx 

    for attempt in range(1, 6):
        t0 = time.perf_counter()
        try:
            r = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            latency = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            
            content = (data["choices"][0]["message"]["content"] or "").strip()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            if prompt_tokens is None or completion_tokens is None or total_tokens is None:
                prompt_tokens = current_prompt_tokens
                completion_tokens = approx_tokens(content)
                total_tokens = prompt_tokens + completion_tokens

            return content, "", latency, int(prompt_tokens), int(completion_tokens), int(total_tokens)
            
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            server_response_text = ""
            
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    last_err += f" | Server Response: {json.dumps(error_details)}"
                    server_response_text = error_details.get("error", {}).get("message", "")
                except json.JSONDecodeError:
                    server_response_text = e.response.text
                    last_err += f" | Server Response (text): {server_response_text[:200]}"
            
            print(f"    -> API Error (Attempt {attempt}): {last_err}")

            # Context length error handling
            if "maximum context length" in server_response_text:
                match = re.search(r'\((\d+) in the messages,', server_response_text)
                if match:
                    server_prompt_tokens = int(match.group(1))
                    print(f"    -> Context Error Detected: Server says prompt is {server_prompt_tokens} tokens (our approx was {current_prompt_tokens}).")
                    
                    new_available = MODEL_MAX_CONTEXT - server_prompt_tokens - TOKEN_BUFFER
                    
                    if new_available <= 0:
                        err_msg = f"Fatal Context Error: Server prompt tokens ({server_prompt_tokens}) exceed model limit."
                        print(f"    -> {err_msg}")
                        return "", err_msg, None, server_prompt_tokens, 0, server_prompt_tokens

                    new_max_tokens = min(DEFAULT_MAX_COMPLETION, new_available)
                    payload["max_tokens"] = new_max_tokens
                    print(f"    -> Retrying with new max_tokens: {new_max_tokens}")
                    current_prompt_tokens = server_prompt_tokens
                else:
                    print("    -> Context Error Detected, but couldn't parse. Sleeping.")

            time.sleep(min(2**attempt, 30))
            
    return "", last_err, None, None, None, None

# ============= Progress Management =============
def load_progress():
    """이전 진행상황 로드"""
    if os.path.exists(PROGRESS_CSV):
        try:
            df = pd.read_csv(PROGRESS_CSV, dtype=str, low_memory=False)
            print(f"✓ 이전 진행상황 발견: {len(df)} 행 이미 처리됨")
            return df
        except Exception as e:
            print(f"⚠ 진행상황 로드 실패: {e}")
    return pd.DataFrame()

def save_progress(df, is_final=False):
    """중간 진행상황 저장"""
    try:
        df.to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")
        if is_final:
            # 최종 결과는 OUTPUT_CSV에도 저장
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"✓ 최종 결과 저장: {OUTPUT_CSV}")
        else:
            print(f"  → 중간 저장 완료 ({len(df)} 행)")
    except Exception as e:
        print(f"⚠ 저장 실패: {e}")

def get_processed_indices(progress_df):
    """이미 처리된 행의 인덱스 집합 반환"""
    if progress_df.empty:
        return set()
    if 'original_index' in progress_df.columns:
        return set(progress_df['original_index'].astype(int))
    return set()

# ============= Main Processing =============
def main():
    # Create output directory
    output_dir = Path(OUTPUT_CSV).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"▶ 모델: {TOGETHER_MODEL}")
    print(f"▶ 입력 파일: {INPUT_CSV}")
    print(f"▶ 출력 파일: {OUTPUT_CSV}")
    print(f"▶ 중간 저장: 매 {SAVE_INTERVAL} 행마다\n")
    
    # Load input CSV
    print("⏳ CSV 파일 로딩 중...")
    df = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False)
    total_rows = len(df)
    print(f"✓ 총 {total_rows:,} 행 로드됨\n")
    
    # Filter for hallucinated packages
    print("🔍 환각 패키지가 있는 행 필터링 중...")
    hallucinated_rows = []
    for idx, row in df.iterrows():
        if idx % 10000 == 0 and idx > 0:
            print(f"  → 진행 중: {idx:,}/{total_rows:,} 행 확인됨...")
        
        has_hallucination, packages = has_hallucinated_packages_from_csv(row)
        if has_hallucination:
            row_dict = row.to_dict()
            row_dict['original_index'] = idx
            row_dict['hallucinated_packages'] = packages
            hallucinated_rows.append(row_dict)
    
    hallucinated_df = pd.DataFrame(hallucinated_rows)
    hallucinated_count = len(hallucinated_df)
    
    print(f"\n✓ 환각 패키지가 있는 행: {hallucinated_count:,} / {total_rows:,} ({hallucinated_count/total_rows*100:.2f}%)\n")
    
    if hallucinated_count == 0:
        print("⚠ 환각 패키지가 있는 행이 없습니다. 종료합니다.")
        return
    
    # Show sample rows
    print("📋 샘플 (처음 3개 행):")
    for i, row in hallucinated_df.head(3).iterrows():
        packages = row['hallucinated_packages']
        print(f"  Row {row['original_index']}: {len(packages)} packages - {packages[:3]}{'...' if len(packages) > 3 else ''}")
    
    # User confirmation
    print(f"\n📊 {hallucinated_count:,}개 행에 대해 self-refinement를 실행합니다.")
    user_input = input("계속하시겠습니까? (y/n): ").strip().lower()
    if user_input != 'y':
        print("❌ 사용자가 취소했습니다.")
        return
    
    # Load previous progress
    progress_df = load_progress()
    processed_indices = get_processed_indices(progress_df)
    
    # gpt_oss의 경우 refinement_response가 NaN인 행들을 재처리
    if MODEL_NAME == "gpt_oss" and not progress_df.empty:
        nan_count = 0
        if 'refinement_response' in progress_df.columns:
            nan_mask = progress_df['refinement_response'].isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                # NaN인 행들의 인덱스를 processed_indices에서 제거
                nan_indices = progress_df[nan_mask]['original_index'].astype(int).tolist()
                processed_indices = processed_indices - set(nan_indices)
                print(f"⚠️  gpt_oss: {nan_count}개 NaN 응답 발견 → 재처리 예정")
    
    if processed_indices:
        print(f"✓ {len(processed_indices):,} 행 건너뛰기")
    
    # Statistics
    latencies = []
    prompt_tok_sum = 0
    compl_tok_sum = 0
    total_tok_sum = 0
    processed_count = len(processed_indices)
    
    out_rows = progress_df.to_dict('records') if not progress_df.empty else []
    
    t_begin = time.perf_counter()
    
    # Process each hallucinated row
    for idx, row in hallucinated_df.iterrows():
        original_idx = row['original_index']
        
        # Skip if already processed
        if original_idx in processed_indices:
            continue
        
        processed_count += 1
        packages = row['hallucinated_packages']
        
        # Create refinement prompt
        system_prompt = "You are an expert in npm package verification."
        user_prompt = create_refinement_prompt(packages)
        
        # Call Together AI with retry logic for empty responses
        resp, err, latency, ptok, ctok, ttok = "", "", None, None, None, None
        retry_count = 0
        
        while retry_count < MAX_RETRIES:
            resp, err, latency, ptok, ctok, ttok = together_chat(system_prompt, user_prompt)
            
            # 응답이 비어있고 에러도 없으면 재시도
            if not resp and not err and retry_count < MAX_RETRIES - 1:
                retry_count += 1
                print(f"    ⚠️  빈 응답 감지 (재시도 {retry_count}/{MAX_RETRIES-1}). {RETRY_DELAY}초 대기 중...")
                time.sleep(RETRY_DELAY)
                continue
            
            # 응답이 있거나 에러가 있거나 마지막 재시도면 중단
            break
        
        # Log progress
        if latency is not None:
            latencies.append(latency)
            avg = sum(latencies) / len(latencies)
            progress_pct = (processed_count / hallucinated_count * 100)
            
            print(f"[{processed_count:,}/{hallucinated_count:,}] ({progress_pct:.1f}%) | "
                  f"{latency:.2f}s | 평균 {avg:.2f}s" + 
                  (f" | ERR: {err}" if err else ""))
        else:
            print(f"[{processed_count:,}/{hallucinated_count:,}] ERR: {err}")
        
        if ptok is not None: prompt_tok_sum += ptok
        if ctok is not None: compl_tok_sum += ctok
        if ttok is not None: total_tok_sum += ttok
        
        # Create output row
        out = {k: row.get(k, "") for k in hallucinated_df.columns}
        out["refinement_prompt"] = user_prompt
        out["refinement_response"] = resp
        out["refinement_error"] = err
        out["refinement_latency_sec"] = f"{latency:.4f}" if latency is not None else ""
        out["refinement_prompt_tokens"] = ptok if ptok is not None else ""
        out["refinement_completion_tokens"] = ctok if ctok is not None else ""
        out["refinement_total_tokens"] = ttok if ttok is not None else ""
        out_rows.append(out)
        
        # Save progress
        if processed_count % SAVE_INTERVAL == 0:
            save_progress(pd.DataFrame(out_rows), is_final=False)
    
    # Final save
    out_df = pd.DataFrame(out_rows)
    save_progress(out_df, is_final=True)
    
    # Print statistics
    t_elapsed = time.perf_counter() - t_begin
    avg_sec = (sum(latencies) / len(latencies)) if latencies else 0
    
    print("\n========== 결과 요약 ==========")
    print(f"- 입력 토큰 합계               : {prompt_tok_sum:,}")
    print(f"- 출력 토큰 합계               : {compl_tok_sum:,}")
    print(f"- 총 토큰 합계                 : {total_tok_sum:,}")
    print(f"- 처리된 행 수                 : {processed_count:,} / {hallucinated_count:,}")
    print(f"- 평균 처리 시간               : {avg_sec:.2f} 초/행")
    print(f"- 총 소요 시간                 : {t_elapsed/3600:.2f} 시간 ({t_elapsed/60:.1f}분)")
    print(f"- 결과 저장                    : {OUTPUT_CSV}")
    print("================================")

if __name__ == "__main__":
    main()
