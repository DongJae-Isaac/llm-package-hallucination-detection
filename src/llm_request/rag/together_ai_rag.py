#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
together_ai_rag.py

RAG(Retrieval-Augmented Generation) 기반 코드 생성 스크립트:
1) npm 패키지 지식베이스(KB)에서 관련 패키지 검색(Retrieval)
2) 검색된 실제 패키지만 사용하도록 Together AI 모델에게 제공
3) 환각 방지를 위해 KB의 패키지만 사용하도록 강제

Requirements:
  - data/npm_package_names.csv (npm package KB)
  
Usage:
  python together_ai_rag.py <model_name> <together_model> <api_key_num>
  
  Example:
    python together_ai_rag.py gemma "google/gemma-3n-E4B-it" 1
"""

import os
import sys
import time
import difflib
import re
import json
import math
import numpy as np
from pathlib import Path
from typing import Optional, List

import requests
import pandas as pd

# No external search dependencies needed (simple retrieval)


# ============= API Keys =============
API_KEYS = {
    "1": "",
    "2": "",
    "3": "",
    "4": "",
}


# ============= Parse Command Line Arguments =============
import argparse

parser = argparse.ArgumentParser(
    description='RAG-based code generation with Together AI',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  python together_ai_rag.py gemma "google/gemma-3n-E4B-it" 1
  python together_ai_rag.py mistral "mistralai/Mistral-7B-Instruct-v0.2" 2
    """
)

parser.add_argument('model_name', type=str,
                   help='Model identifier (e.g., gemma, gpt_oss, marin, mistral, qwen, codellama7b)')
parser.add_argument('together_model', type=str,
                   help='Together AI model name (e.g., "google/gemma-3n-E4B-it")')
parser.add_argument('api_key_num', type=str, choices=['1', '2', '3', '4'],
                   help='API key number (1, 2, 3, or 4)')

args = parser.parse_args()

MODEL_NAME = args.model_name
TOGETHER_MODEL = args.together_model
API_KEY_NUM = args.api_key_num

if API_KEY_NUM not in API_KEYS:
    print(f"❌ Invalid API key number: {API_KEY_NUM}. Must be 1, 2, 3, or 4.")
    sys.exit(1)

API_KEY = API_KEYS[API_KEY_NUM]


# ============= Configuration =============
INPUT_CSV = f"result/result_paper_prompts_expanded_v2_out_{MODEL_NAME}_final.csv"
OUTPUT_CSV = f"result/rag/result_rag_{MODEL_NAME}.csv"
PROGRESS_CSV = f"progress_rag_{MODEL_NAME}.csv"

# RAG KB paths
KB_CSV = "data/npm_package_names.csv"
KB_METADATA_CSV = "data/npm_package_metadata.csv"  # Optional: name, description, keywords

CHUNK_SIZE = 1000
BASE_URL = "https://api.together.xyz/v1/chat/completions"

# Model-specific timeout settings
if MODEL_NAME == "gpt_oss":
    TIMEOUT = 600  # 10 minutes
    print(f"⚙️  gpt_oss 모드: 타임아웃 {TIMEOUT}초")
else:
    TIMEOUT = 300  # 5 minutes

SAVE_INTERVAL = 100

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


# ---------------------------------------------------------------------------
# RAG Knowledge Base
# ---------------------------------------------------------------------------

# Global variables for simple search
_kb_names_cache = None

def load_pkg_kb(kb_csv: str, use_metadata: bool = False) -> List[str]:
    """
    Load npm package knowledge base from CSV.
    Simple loading without BM25 indexing (ollama_codellama_rag.py style).
    Expected format: CSV with 'name' column containing package names.
    
    Args:
        kb_csv: Path to package names CSV
        use_metadata: Ignored (for compatibility)
    """
    global _kb_names_cache
    
    try:
        # Load package names
        df = pd.read_csv(kb_csv, dtype=str, low_memory=False)
        if "name" in df.columns:
            col = "name"
        else:
            col = df.columns[0]
        names = df[col].astype(str).str.strip().str.lower().tolist()
        names = [n for n in names if n and n != 'nan']
        
        _kb_names_cache = names
        
        return names
    except Exception as e:
        print(f"⚠ KB loading error: {e}")
        return []





def retrieve_pkg_candidates(question: str, kb_names: List[str], top_k: int = 20) -> List[str]:
    """
    Fast RAG Retrieval with early stopping.
    Token matching + string similarity scoring.
    Optimized for large KB (3.6M+ packages).
    """
    if not isinstance(question, str) or not question.strip():
        return []

    q_lower = question.lower()

    # Extract alphanumeric tokens from question (minimum 3 chars)
    tokens = re.findall(r"[a-z0-9_]+", q_lower)
    tokens = [t for t in tokens if len(t) >= 3]
    
    if not tokens:
        return []

    scores = {}
    checked = 0
    max_check = 50000  # 최대 5만개만 검사 (366만개 → 5만개)

    for name in kb_names:
        n = name.lower()
        score = 0.0

        # 1) Token matching: boost if tokens appear in package name
        matched = False
        for t in tokens:
            if t in n:
                score += 2.0
                matched = True

        # 2) String similarity only for matched packages
        if matched:
            score += difflib.SequenceMatcher(None, n, q_lower).ratio()
            scores[name] = score

        checked += 1
        
        # Early stopping: 충분한 매칭을 찾거나 최대 검사량 도달
        if len(scores) >= 200 or checked >= max_check:
            break

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


# ============= Together AI API Call with RAG =============
def together_chat(system_prompt: str, user_prompt: str, kb_candidates: Optional[List[str]] = None):
    """
    Together chat.completions 호출 + RAG context 주입
    Returns: (content, error, latency, prompt_tokens, completion_tokens, total_tokens)
    """
    
    # 1) Add RAG context to system prompt
    rag_context = ""
    if kb_candidates:
        pkg_lines = "\n".join(f"- {name}" for name in kb_candidates)
        rag_context = f"""

IMPORTANT: Below is a list of REAL npm packages retrieved from a trusted knowledge base.
You MUST use only the packages in the following list.
Do NOT invent or hallucinate new package names under any circumstances.

[Allowed npm packages]
{pkg_lines}

If none of these packages are appropriate for the task,
clearly state: "No appropriate package found in the allowed list."

Do not write or reference any npm package that is not in the list above.
"""

    # 2) Combine original system prompt + RAG constraint
    final_system_prompt = (system_prompt or "") + rag_context
    
    # 3) Calculate tokens for context length management
    approx_input = f"[SYSTEM]\n{final_system_prompt}\n[/SYSTEM]\n[USER]\n{user_prompt or ''}\n[/USER]"
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
            {"role": "system", "content": final_system_prompt},
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
def has_hallucinated_packages_from_csv(row):
    """
    CSV의 response_npm_hallucinated 컬럼에서 환각 패키지 확인
    Returns: (has_hallucination: bool, packages: list)
    """
    try:
        hallucinated_str = row.get('response_npm_hallucinated', '')
        
        if pd.isna(hallucinated_str) or hallucinated_str == '' or hallucinated_str == '[]':
            return False, []
        
        import ast
        packages = ast.literal_eval(hallucinated_str)
        
        if isinstance(packages, list) and len(packages) > 0:
            return True, packages
        
        return False, []
    except Exception as e:
        return False, []


def main():
    # Create output directory
    output_dir = Path(OUTPUT_CSV).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"▶ 모델: {TOGETHER_MODEL}")
    print(f"▶ 입력 파일: {INPUT_CSV}")
    print(f"▶ 출력 파일: {OUTPUT_CSV}")
    print(f"▶ 중간 저장: 매 {SAVE_INTERVAL} 행마다\n")
    
    # Load package KB for RAG
    print("⏳ npm 패키지 KB 로딩 중...")
    kb_names = load_pkg_kb(KB_CSV)
    if kb_names:
        print(f"✓ {len(kb_names):,}개 패키지 로드됨\n")
    else:
        print("⚠ KB 로딩 실패. RAG 없이 진행합니다.\n")
    
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
    print(f"\n📊 {hallucinated_count:,}개 행에 대해 RAG 코드 생성을 실행합니다.")
    user_input = input("계속하시겠습니까? (y/n): ").strip().lower()
    if user_input != 'y':
        print("❌ 사용자가 취소했습니다.")
        return
    
    # Load previous progress
    progress_df = load_progress()
    processed_indices = get_processed_indices(progress_df)
    
    if processed_indices:
        print(f"✓ {len(processed_indices):,} 행 건너뛰기\n")
    
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
        
        sys_p = row.get("system_prompt", "") or ""
        usr_p = row.get("request_prompt", "") or ""
        
        # RAG: Retrieve relevant packages from KB based on hallucinated packages + prompt
        query_text = usr_p + " " + " ".join(packages)
        kb_candidates = retrieve_pkg_candidates(query_text, kb_names, top_k=20)
        
        # Call Together AI with RAG context
        resp, err, latency, ptok, ctok, ttok = together_chat(sys_p, usr_p, kb_candidates)
        
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
        out["rag_response"] = resp
        out["rag_error"] = err
        out["rag_latency_sec"] = f"{latency:.4f}" if latency is not None else ""
        out["rag_prompt_tokens"] = ptok if ptok is not None else ""
        out["rag_completion_tokens"] = ctok if ctok is not None else ""
        out["rag_total_tokens"] = ttok if ttok is not None else ""
        out["rag_candidates"] = ",".join(kb_candidates)
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
