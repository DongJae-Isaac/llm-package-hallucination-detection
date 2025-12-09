# 전체 프롬프트 실행
# 중간 저장 기능(10행마다) - SAVE_INTERVAL 변수

import os, time, sys, json, math, re
import pandas as pd
import requests


INPUT_CSV   = r"D:\Programming\@@Sec\paper_prompts_expanded_v2.csv"
OUTPUT_CSV  = r"D:\Programming\@@Sec\paper_prompts_expanded_v2_out_marin_final.csv"
PROGRESS_CSV = r"D:\Programming\@@Sec\progress_checkpoint_marin.csv" 
MODEL_NAME  = "marin-community/marin-8b-instruct"       
#SAMPLE_N    = 10
CHUNK_SIZE = 100
BASE_URL    = "https://api.together.xyz/v1/chat/completions"
TIMEOUT     = 120
SAVE_INTERVAL = 10

API_KEY = ""

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


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

def count_rows_precisely(csv_path: str, chunksize: int = 200_000) -> int:
    total = 0
    for ch in pd.read_csv(csv_path, dtype=str, low_memory=False, chunksize=chunksize):
        total += len(ch)
    return total

# 모델의 최대 컨텍스트 길이 (오류 메시지에서 확인)
MODEL_MAX_CONTEXT = 4096 
# API의 기본 최대 출력 토큰 (오류 메시지에서 확인)
DEFAULT_MAX_COMPLETION = 2048
# 토큰 근사 계산 오차를 대비한 여유분(버퍼)
TOKEN_BUFFER = 100 # <--- 오차를 감안해 16보다 넉넉하게 100으로 설정

def together_chat(system_prompt: str, user_prompt: str):
    """Together chat.completions 호출 + usage/latency 반환 (오류 파싱 및 재시도)"""
    
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
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user",   "content": user_prompt or ""},
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": final_max_tokens # 첫 시도 값
    }
    
    last_err = ""
    # `prompt_tokens_approx`는 재시도 시 서버 값으로 덮어써질 수 있음
    current_prompt_tokens = prompt_tokens_approx 

    for attempt in range(1, 6):
        t0 = time.perf_counter()
        try:
            # 재시도 시, payload["max_tokens"]는 수정된 값일 수 있음
            r = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
            latency = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()
            
            content = (data["choices"][0]["message"]["content"] or "").strip()
            usage = data.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

            # usage가 없으면, 우리가 알던 최신 토큰 값(current_prompt_tokens)을 사용
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

            # --- [!] 핵심 로직: 오류 메시지 파싱 및 재시도 ---
            if "maximum context length" in server_response_text:
                # 정규표현식으로 서버가 알려준 "진짜" 토큰 값을 찾음
                match = re.search(r'\((\d+) in the messages,', server_response_text)
                if match:
                    server_prompt_tokens = int(match.group(1)) # 예: 2311
                    print(f"    -> Context Error Detected: Server says prompt is {server_prompt_tokens} tokens (our approx was {current_prompt_tokens}).")
                    
                    # "진짜" 토큰 값으로 max_tokens를 다시 계산
                    new_available = MODEL_MAX_CONTEXT - server_prompt_tokens - TOKEN_BUFFER
                    
                    if new_available <= 0:
                        err_msg = f"Fatal Context Error: Server prompt tokens ({server_prompt_tokens}) exceed model limit."
                        print(f"    -> {err_msg}")
                        return "", err_msg, None, server_prompt_tokens, 0, server_prompt_tokens # 재시도 중단

                    new_max_tokens = min(DEFAULT_MAX_COMPLETION, new_available)
                    
                    # [!] 다음 루프(재시도)를 위해 payload의 max_tokens 값을 덮어씁니다.
                    payload["max_tokens"] = new_max_tokens
                    print(f"    -> Retrying with new max_tokens: {new_max_tokens}")
                    
                    # 다음에 usage가 안 나올 경우를 대비해, "진짜" 토큰 값으로 업데이트
                    current_prompt_tokens = server_prompt_tokens
                    
                else:
                    print("    -> Context Error Detected, but couldn't parse. Sleeping.")
            # --- [수정 끝] ---

            time.sleep(min(2**attempt, 30))
            
    return "", last_err, None, None, None, None

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
    # 원본 행 번호가 있다면 사용, 없으면 빈 집합
    if 'original_index' in progress_df.columns:
        return set(progress_df['original_index'].astype(int))
    return set()
        
def main():
    # 총 행 수(정보용)
    try:
        total_rows = count_rows_precisely(INPUT_CSV)
    except Exception:
        total_rows = 0

    # 이전 진행상황 로드
    progress_df = load_progress()
    processed_indices = get_processed_indices(progress_df)
    if processed_indices:
        print(f"✓ {len(processed_indices):,} 행 건너뛰기")
        remaining = total_rows - len(processed_indices) if total_rows else "알 수 없음"
        print(f"  → 남은 행: {remaining:,}" if isinstance(remaining, int) else f"  → 남은 행: {remaining}")
    
    # 통계 변수
    latencies = []
    prompt_tok_sum = 0
    compl_tok_sum = 0
    total_tok_sum = 0
    processed_count = len(processed_indices)

    print(f"▶ 모델: {MODEL_NAME}")
    print(f"▶ 입력 파일: {INPUT_CSV}")
    if total_rows:
        print(f"▶ 총 행 수(정확): {total_rows:,}")
    # print(f"▶ 샘플 크기: {SAMPLE_N}")
    print(f"▶ 청크 크기: {CHUNK_SIZE} 행")
    print(f"▶ 중간 저장: 매 {SAVE_INTERVAL} 행마다\n")

    t_begin = time.perf_counter()
    out_rows = progress_df.to_dict('records') if not progress_df.empty else []
    
    # 청크 단위로 읽기
    for chunk_num, chunk_df in enumerate(pd.read_csv(INPUT_CSV, dtype=str, low_memory=False, chunksize=CHUNK_SIZE), start=1):
        # 필수 컬럼 확인
        for c in ["system_prompt","request_prompt","system_prompt_type","question_num","question_t_num"]:
            if c not in chunk_df.columns:
                chunk_df[c] = ""
        
        # 원본 인덱스 추가
        chunk_df['original_index'] = chunk_df.index + (chunk_num - 1) * CHUNK_SIZE
        
        for idx, row in chunk_df.iterrows():
            original_idx = row['original_index']
            
            # 이미 처리된 행은 건너뛰기
            if original_idx in processed_indices:
                continue
            
            processed_count += 1
            sys_p = row.get("system_prompt", "") or ""
            usr_p = row.get("request_prompt", "") or ""

            resp, err, latency, ptok, ctok, ttok = together_chat(sys_p, usr_p)

            # 진행 로그
            if latency is not None:
                latencies.append(latency)
                avg = sum(latencies) / len(latencies)
                progress_pct = (processed_count / total_rows * 100) if total_rows else 0
                
                print(f"[{processed_count:,}/{total_rows:,}] ({progress_pct:.1f}%) | "
                      f"{latency:.2f}s | 평균 {avg:.2f}s" + 
                      (f" | ERR: {err}" if err else ""))
            else:
                print(f"[{processed_count:,}/{total_rows:,}] ERR: {err}")

            if ptok is not None: prompt_tok_sum += ptok
            if ctok is not None: compl_tok_sum += ctok
            if ttok is not None: total_tok_sum += ttok

            # 결과 행 생성
            out = {k: row.get(k, "") for k in chunk_df.columns}
            out["response_prompt"] = resp
            out["error"] = err
            out["latency_sec"] = f"{latency:.4f}" if latency is not None else ""
            out["prompt_tokens"] = ptok if ptok is not None else ""
            out["completion_tokens"] = ctok if ctok is not None else ""
            out["total_tokens"] = ttok if ttok is not None else ""
            out_rows.append(out)

            # 중간 저장
            if processed_count % SAVE_INTERVAL == 0:
                save_progress(pd.DataFrame(out_rows), is_final=False)
    

    # 최종 저장
    out_df = pd.DataFrame(out_rows)
    save_progress(out_df, is_final=True)
    
    # 통계 출력
    t_elapsed = time.perf_counter() - t_begin
    avg_sec = (sum(latencies) / len(latencies)) if latencies else 0

    print("\n========== 결과 요약 ==========")
    print(f"- 입력 토큰 합계 (샘플)          : {prompt_tok_sum:,}")
    print(f"- 출력 토큰 합계 (샘플)          : {compl_tok_sum:,}")
    print(f"- 총 토큰 합계 (샘플)            : {total_tok_sum:,}")
    print(f"- 처리된 행 수              : {processed_count:,} / {total_rows:,}")
    print(f"- 평균 처리 시간            : {avg_sec:.2f} 초/행")
    print(f"- 총 소요 시간              : {t_elapsed/3600:.2f} 시간 ({t_elapsed/60:.1f}분)")
    print(f"- 결과 저장                      : {OUTPUT_CSV}")
    print("================================")

if __name__ == "__main__":
    main()
