import os
import time
import json
import re
import difflib
from typing import List, Optional

import pandas as pd
import requests

# ===========================
# 경로 설정 (환경에 맞게 수정)
# ===========================
INPUT_CSV    = r"D:\Programming\@@Sec\paper_prompts_expanded_v2.csv"
OUTPUT_CSV   = r"D:\Programming\@@Sec\result_paper_prompts_rag_codellama7b.csv"
PROGRESS_CSV = r"D:\Programming\@@Sec\progress_paper_prompts_rag_codellama7b.csv"
KB_CSV       = r"D:\Programming\@@Sec\npm_package_names.csv"   # npm 패키지 이름 목록

# ===========================
# 실행 파라미터
# ===========================
CHUNK_SIZE    = 100    # csv를 청크 단위로 읽을 때 크기
SAVE_INTERVAL = 10     # 이 개수만큼 새로 처리할 때마다 중간 저장

# ===========================
# Ollama 설정
# ===========================
MODEL_NAME    = "codellama:7b"
OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 500    # seconds
TEMPERATURE   = 0.0
TOP_P         = 0.9


# ---------------------------------------------------------------------------
# 기본 유틸
# ---------------------------------------------------------------------------
def count_rows_precisely(csv_path: str, chunksize: int = 200_000) -> int:
    """대용량 CSV의 총 행 수를 정확히 세기 위한 함수."""
    total = 0
    for ch in pd.read_csv(csv_path, dtype=str, low_memory=False, chunksize=chunksize):
        total += len(ch)
    return total


def load_progress() -> pd.DataFrame:
    """이전 진행상황(progress CSV)을 불러온다. 없으면 빈 DataFrame."""
    if os.path.exists(PROGRESS_CSV):
        try:
            df = pd.read_csv(PROGRESS_CSV, dtype=str, low_memory=False)
            print(f"✓ 이전 진행상황 발견: {len(df)} 행 이미 처리됨")
            return df
        except Exception as e:
            print(f"⚠ 진행상황 로드 실패: {e}")
    return pd.DataFrame()


def save_progress(df: pd.DataFrame, is_final: bool = False) -> None:
    """중간/최종 진행상황 저장."""
    try:
        df.to_csv(PROGRESS_CSV, index=False, encoding="utf-8-sig")
        if is_final:
            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"✓ 최종 결과 저장: {OUTPUT_CSV}")
        else:
            print(f"  → 중간 저장 완료 ({len(df)} 행)")
    except Exception as e:
        print(f"⚠ 저장 실패: {e}")


def get_processed_indices(progress_df: pd.DataFrame) -> set:
    """이미 처리된 행의 original_index 집합을 반환."""
    if progress_df.empty:
        return set()
    if "original_index" in progress_df.columns:
        return set(progress_df["original_index"].astype(int))
    return set()


# ---------------------------------------------------------------------------
# npm 패키지 KB + RAG Retrieval
# ---------------------------------------------------------------------------
def load_pkg_kb(path: str) -> List[str]:
    """
    npm_package_names.csv 에서 패키지 이름 리스트를 로드.
    - 기본적으로 'name' 컬럼을 사용하되, 없으면 첫 번째 컬럼 사용.
    - 모든 이름은 소문자 / strip 처리 후 리스트로 반환.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if "name" in df.columns:
        col = "name"
    else:
        col = df.columns[0]
    names = df[col].astype(str).str.strip().str.lower().tolist()
    return names


def retrieve_pkg_candidates(question: str, kb_names: List[str], top_k: int = 10) -> List[str]:
    """
    RAG의 Retrieval 부분.
    - 질문 텍스트를 보고 npm 패키지 KB에서 관련 패키지 상위 top_k개를 찾는다.
    - 간단 구현: 토큰 부분 일치 + 문자열 유사도 점수의 합으로 랭킹.
    """
    if not isinstance(question, str) or not question.strip():
        return []

    q_lower = question.lower()

    # 질문에서 알파벳/숫자 토큰 추출 (너무 짧은 건 제외)
    tokens = re.findall(r"[a-z0-9_]+", q_lower)
    tokens = [t for t in tokens if len(t) >= 3]

    scores: dict[str, float] = {}

    for name in kb_names:
        n = name.lower()
        score = 0.0

        # 1) 토큰이 이름에 포함되면 가중치 부여
        for t in tokens:
            if t in n:
                score += 2.0

        # 2) 전체 질문과의 대략적인 문자열 유사도
        score += difflib.SequenceMatcher(None, n, q_lower).ratio()

        if score > 0:
            scores[name] = score

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Ollama 호출 (RAG 컨텍스트 포함)
# ---------------------------------------------------------------------------
def ollama_chat(system_prompt: str,
                user_prompt: str,
                kb_candidates: Optional[List[str]] = None):
    """
    Ollama codellama:7b chat wrapper (RAG version).
    kb_candidates: Candidate npm package names retrieved from the KB.
    If provided, the RAG context will force the model to use ONLY these packages
    and avoid inventing or hallucinating any new ones.
    """
    # 1) RAG context (English)
    rag_context = ""
    if kb_candidates:
        pkg_lines = "\n".join(f"- {name}" for name in kb_candidates)
        rag_context = f"""
You are an assistant that generates JavaScript/Node.js code using REAL npm packages.

Below is a list of candidate packages retrieved from a trusted knowledge base.
You MUST use only the packages in the following list.
Do NOT invent or hallucinate new package names under any circumstances.

[Allowed npm packages]
{pkg_lines}

If none of these packages are appropriate for the task,
clearly answer: "No appropriate package found in the allowed list."

Do not write or reference any npm package that is not in the list above.
"""

    # 2) Combine original system prompt + RAG constraint
    final_system_prompt = (system_prompt or "") + rag_context

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": final_system_prompt},
            {"role": "user",   "content": user_prompt or ""},
        ],
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            # "num_predict": 2048,  # 필요하면 토큰 길이 제한
        },
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        latency = time.perf_counter() - t0
        r.raise_for_status()
        data = r.json()
        # Ollama /api/chat 응답 형식: {"message":{"role":"assistant","content":"..."}, ...}
        content = ((data.get("message") or {}).get("content") or "").strip()
        # 토큰 카운트는 모델 설정에 따라 없을 수도 있어서 기본 0
        prompt_tok = data.get("prompt_eval_count", 0)
        compl_tok = data.get("eval_count", 0)
        total_tok = prompt_tok + compl_tok
        return content, "", latency, int(prompt_tok), int(compl_tok), int(total_tok)
    except Exception as e:
        latency = time.perf_counter() - t0
        err_msg = f"{type(e).__name__}: {e}"
        return "", err_msg, latency, 0, 0, 0


# ---------------------------------------------------------------------------
# 메인 루프 (RAG 기반 코드 생성)
# ---------------------------------------------------------------------------
def main():
    # 1) 전체 행 수 계산 (정보용)
    try:
        total_rows = count_rows_precisely(INPUT_CSV)
    except Exception:
        total_rows = 0

    # 2) 패키지 지식베이스 로드 (RAG용)
    try:
        kb_names = load_pkg_kb(KB_CSV)
        print(f"▶ 패키지 KB 로딩 완료: {len(kb_names):,} 개")
    except Exception as e:
        print(f"⚠ 패키지 KB 로딩 실패: {e}")
        kb_names = []

    # 3) 이전 진행상황 로드
    progress_df = load_progress()
    processed_indices = get_processed_indices(progress_df)
    if processed_indices:
        print(f"✓ {len(processed_indices):,} 행 건너뛰기 (이미 처리됨)")
        if total_rows:
            remaining = total_rows - len(processed_indices)
            print(f"  → 남은 행: {remaining:,}")
        else:
            print("  → 남은 행: 알 수 없음")

    # 4) 통계 변수
    latencies = []
    prompt_tok_sum = 0
    compl_tok_sum = 0
    total_tok_sum = 0
    processed_count = len(processed_indices)

    print(f"▶ 모델: {MODEL_NAME}")
    print(f"▶ 입력 파일: {INPUT_CSV}")
    if total_rows:
        print(f"▶ 총 행 수(정확): {total_rows:,}")
    print(f"▶ 청크 크기: {CHUNK_SIZE} 행")
    print(f"▶ 중간 저장: 매 {SAVE_INTERVAL} 행마다\n")

    t_begin = time.perf_counter()
    out_rows = progress_df.to_dict("records") if not progress_df.empty else []

    # 5) 청크 단위로 CSV 읽으면서, RAG + LLM 호출
    chunk_iter = pd.read_csv(INPUT_CSV, dtype=str, low_memory=False, chunksize=CHUNK_SIZE)

    for chunk_num, chunk_df in enumerate(chunk_iter, start=1):
        # 필수 컬럼이 없으면 빈 문자열로 채우기
        for c in ["system_prompt", "request_prompt", "system_prompt_type",
                  "question_num", "question_t_num"]:
            if c not in chunk_df.columns:
                chunk_df[c] = ""

        # 원본 인덱스 부여 (0-based)
        chunk_df["original_index"] = chunk_df.index + (chunk_num - 1) * CHUNK_SIZE

        for _, row in chunk_df.iterrows():
            original_idx = int(row["original_index"])

            # 이미 처리된 행이면 스킵
            if original_idx in processed_indices:
                continue

            processed_count += 1
            sys_p = row.get("system_prompt", "") or ""
            usr_p = row.get("request_prompt", "") or ""

            # 5-1) RAG retrieval: 질문 기반으로 패키지 후보 검색
            kb_candidates = retrieve_pkg_candidates(usr_p, kb_names, top_k=20)

            # 5-2) Ollama 호출 (RAG 컨텍스트 포함)
            resp, err, latency, ptok, ctok, ttok = ollama_chat(
                sys_p, usr_p, kb_candidates=kb_candidates
            )

            # 5-3) 진행 로그
            if latency is not None:
                latencies.append(latency)
                avg = sum(latencies) / len(latencies)
                progress_pct = (processed_count / total_rows * 100) if total_rows else 0.0
                print(
                    f"[{processed_count:,}/{total_rows:,}] ({progress_pct:.1f}%) | "
                    f"{latency:.2f}s | 평균 {avg:.2f}s"
                    + (f" | ERR: {err}" if err else "")
                )
            else:
                print(f"[{processed_count:,}/{total_rows:,}] ERR: {err}")

            prompt_tok_sum += ptok
            compl_tok_sum += ctok
            total_tok_sum += ttok

            # 5-4) 결과 행 구성
            out = {k: row.get(k, "") for k in chunk_df.columns}
            out["response_prompt"] = resp
            out["error"] = err
            out["latency_sec"] = f"{latency:.4f}" if latency is not None else ""
            out["prompt_tokens"] = ptok
            out["completion_tokens"] = ctok
            out["total_tokens"] = ttok

            # 디버깅/분석용: 이번 질의에서 사용한 RAG 후보 패키지 목록
            out["rag_candidates"] = ",".join(kb_candidates)

            out_rows.append(out)

            # 5-5) 중간 저장
            if processed_count % SAVE_INTERVAL == 0:
                save_progress(pd.DataFrame(out_rows), is_final=False)

    # 6) 최종 저장
    out_df = pd.DataFrame(out_rows)
    save_progress(out_df, is_final=True)

    # 7) 통계 출력
    t_elapsed = time.perf_counter() - t_begin
    avg_sec = (sum(latencies) / len(latencies)) if latencies else 0.0

    print("\n========== 결과 요약 ==========")
    print(f"- 처리된 행 수              : {processed_count:,} / {total_rows:,}")
    print(f"- 평균 처리 시간            : {avg_sec:.2f} 초/행")
    print(f"- 총 소요 시간              : {t_elapsed/3600:.2f} 시간 ({t_elapsed/60:.1f}분)")
    print(f"- 총 prompt tokens          : {prompt_tok_sum:,}")
    print(f"- 총 completion tokens      : {compl_tok_sum:,}")
    print(f"- 총 tokens                 : {total_tok_sum:,}")
    print(f"- 결과 저장                 : {OUTPUT_CSV}")
    print("================================")


if __name__ == "__main__":
    main()
