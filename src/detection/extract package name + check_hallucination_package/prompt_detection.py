import os, re, sys, time, json, requests, pandas as pd
from ast import literal_eval
from collections import defaultdict, Counter
from typing import Optional, List, Set, Dict, Callable, Tuple

# -----------------------------
# 기본 설정(무인자 실행)
# -----------------------------
INPUT_FILE  = "D:\slopsquating\sc_projectspaper_prompts_expanded_v2_out.csv"
OUTPUT_FILE = "D:\slopsquating\FINAL_verified_npm_by_system.csv"
SYSTEM_COL  = "system_prompt"
RESPONSE_COL= "response_prompt"
SAVE_INTERVAL = 50
TIMEOUT = 5
RATE_SLEEP = 0.0
RESUME = True

# -----------------------------
# 필터 목록
# -----------------------------
NODE_BUILTINS: Set[str] = {
    'assert','buffer','child_process','cluster','console','crypto','dns','events','fs',
    'http','https','module','net','os','path','process','querystring','readline','stream',
    'string_decoder','timers','tls','tty','url','util','v8','vm','zlib'
}
JS_KEYWORDS: Set[str] = {
    'Promise','fetch','target','event','console','JSON','Date','Math','Array','Object',
    'String','Number','Boolean','Map','Set','Symbol','Intl','WeakMap','WeakSet'
}

# -----------------------------
# 유틸 함수 (정규화/검증/저장)
# -----------------------------
def normalize_token_common(token: str) -> Optional[str]:
    if token is None: return None
    if not isinstance(token, str): token = str(token)
    t = token.strip().strip('`\'"')
    if not t or len(t) > 214: return None
    if t.isdigit() or ' ' in t: return None
    if any(ch in t for ch in '();[]{}<>'): return None
    if t.startswith('$') or t.lower() in {'none','null','n/a','na','nil'}: return None
    return t

def normalize_npm_name(name: str) -> Tuple[Optional[str], bool]:
    """
    npm 패키지명 정제(완화 버전):
      - 허용 문자만 유지: [a-zA-Z0-9@._/\-]
      - 루트 패키지 추출: '@scope/name/sub'→'@scope/name', 'foo/bar'→'foo'
      - '@scope' 단독도 보존(키워드 자체는 유지), 단 BAD_SCOPE=True 플래그로 반환
      - 최종 소문자화
    반환: (normalized, bad_scope_flag)
    """
    t = normalize_token_common(name)
    if t is None: return (None, False)

    if re.search(r'[^a-zA-Z0-9@._/\-]', t):
        return (None, False)

    bad_scope = False
    if t.startswith('@'):
        parts = t.split('/')
        if len(parts) >= 2:
            t = f"{parts[0]}/{parts[1]}"
        else:
            # @scope 단독 → 보존하되 BAD_SCOPE(True)
            bad_scope = True
            # 보존은 하되 npm 존재검증은 실패할 것이므로 그대로 둠
    else:
        t = t.split('/')[0]

    return (t.lower(), bad_scope)

def strip_codeblocks(text: str) -> str:
    return re.sub(r'```.*?```', '', text, flags=re.S)

def parse_line_list(x) -> List[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)): return []
    s = str(x).strip()
    # try python-literal list
    if s.startswith('[') and s.endswith(']'):
        try:
            vals = literal_eval(s)
            if isinstance(vals, list):
                return [int(v) for v in vals if str(v).isdigit()]
        except Exception:
            pass
    # fallback: digits
    return [int(v) for v in re.findall(r'\d+', s)]

def save_csv(df: pd.DataFrame, path: str):
    # list/set → JSON스러운 문자열로 저장
    out = df.copy()
    out['line_numbers'] = out['line_numbers'].apply(lambda xs: list(xs) if isinstance(xs, (list,set,tuple)) else xs)
    out.to_csv(path, index=False, encoding="utf-8-sig")

# -----------------------------
# 추출 전략
# -----------------------------
def strat_comma(text: str) -> List[str]:
    text = strip_codeblocks(text).replace('\n', ',')
    return [t.strip() for t in text.split(',') if t.strip()]

def strat_newline(text: str) -> List[str]:
    text = strip_codeblocks(text)
    return [t.strip() for t in text.splitlines() if t.strip()]

def strat_json(text: str) -> List[str]:
    text = strip_codeblocks(text)
    m = re.search(r'\[.*?\]', text, flags=re.S)
    if not m: return []
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [str(x) for x in arr]
    except Exception:
        return []
    return []

def strat_import(text: str) -> List[str]:
    text = strip_codeblocks(text)
    cands = []
    for m in re.finditer(r"require\(\s*['\"]([^'\"\)]+)['\"]\s*\)", text): cands.append(m.group(1))
    for m in re.finditer(r"from\s+['\"]([^'\"\)]+)['\"]", text): cands.append(m.group(1))
    for m in re.finditer(r"\bimport\s+['\"]([^'\"\)]+)['\"]", text): cands.append(m.group(1))
    for m in re.finditer(r"\bimport\s+([@\w./-]+)\s+from\b", text): cands.append(m.group(1))
    return cands

def strat_npm_install(text: str) -> List[str]:
    text = strip_codeblocks(text)
    cands = []
    for line in text.splitlines():
        if re.search(r'\bnpm\s+(?:i|install)\b', line):
            parts = re.split(r'\s+', line.strip())
            try:
                idx = next(i for i, p in enumerate(parts) if p in {'i','install'})
            except StopIteration:
                continue
            for tok in parts[idx+1:]:
                if not tok.startswith('-') and tok:
                    cands.append(tok)
    return cands

def strat_bullet(text: str) -> List[str]:
    text = strip_codeblocks(text)
    cands = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*•]|[\d]{1,3}[\).])\s*([@\w./-]+)", line.strip())
        if m: cands.append(m.group(1))
    return cands

def strat_fallback(text: str) -> List[str]:
    text = strip_codeblocks(text).replace('\n', ',')
    return [t.strip() for t in text.split(',') if t.strip()]

# -----------------------------
# system_prompt → 전략 선택(보강)
# -----------------------------
STRATEGY_KEYS = [
    # JSON/배열 지시
    ('json', strat_json),
    ('return array', strat_json),
    ('array of', strat_json),
    ('array', strat_json),

    # 쉼표 지시
    ('comma-separated', strat_comma),
    ('comma separated', strat_comma),
    (', separated', strat_comma),
    ('comma', strat_comma),

    # 줄바꿈 지시
    ('one per line', strat_newline),
    ('each on a new line', strat_newline),
    ('new line', strat_newline),
    ('newline', strat_newline),

    # 설치 명령 지시
    ('npm install', strat_npm_install),
    ('use npm i', strat_npm_install),

    # 코드 import/require 지시
    ('import', strat_import),
    ('require', strat_import),

    # 불릿 지시/리스트 지시
    ('bullet', strat_bullet),
    ('list items', strat_bullet),
    ('as a list', strat_bullet),
    ('list', strat_bullet),
]

def choose_strategy(system_prompt: str) -> Tuple[Callable[[str], List[str]], str]:
    sp = (system_prompt or "").lower()
    for key, strat in STRATEGY_KEYS:
        if key in sp:
            return strat, strat.__name__
    return strat_comma, 'strat_comma'

# -----------------------------
# npm Registry 조회
# -----------------------------
def exists_on_npm(pkg: str) -> bool:
    try:
        r = requests.get(f"https://registry.npmjs.org/{pkg}", timeout=TIMEOUT)
        return r.status_code == 200
    except requests.RequestException:
        return False

# -----------------------------
# SC → 기대 키워드 매핑(재현용)
# -----------------------------
def build_expected(sc_df: pd.DataFrame) -> Tuple[Dict[str, Set[int]], Dict[str, str], Dict[str, bool]]:
    """
    sc_df를 기준으로 system_prompt 전략을 적용해
    - expected_keywords: kw -> {line_numbers}
    - expected_strategy : kw -> strategy_name(최초)
    - bad_scope_map     : kw -> BAD_SCOPE 여부
    를 구축한다.
    """
    expected_keywords: Dict[str, Set[int]] = defaultdict(set)
    expected_strategy: Dict[str, str] = {}
    bad_scope_map: Dict[str, bool] = {}

    for idx, row in sc_df.iterrows():
        sp = str(row.get(SYSTEM_COL, "")) if SYSTEM_COL in sc_df.columns else ""
        rp = str(row.get(RESPONSE_COL, "")) if RESPONSE_COL in sc_df.columns else ""
        if not rp.strip():
            continue
        fn, strat_name = choose_strategy(sp)
        cands = fn(rp)
        if not cands and fn is not strat_fallback:
            cands = strat_fallback(rp)
            strat_name = 'strat_fallback'
        for raw in cands:
            norm, bad_scope = normalize_npm_name(raw)
            if not norm: continue
            ln = idx + 2  # header 고려(사람 기준 1-based)
            expected_keywords[norm].add(ln)
            expected_strategy.setdefault(norm, strat_name)
            bad_scope_map[norm] = bad_scope_map.get(norm, False) or bad_scope

    return expected_keywords, expected_strategy, bad_scope_map

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"입력 파일({INPUT_FILE})을 찾을 수 없습니다.")
        sys.exit(1)

    sc = pd.read_csv(INPUT_FILE)
    if SYSTEM_COL not in sc.columns or RESPONSE_COL not in sc.columns:
        print(f"'{SYSTEM_COL}', '{RESPONSE_COL}' 열을 찾을 수 없습니다.")
        sys.exit(1)

    # 0) SC 기준 기대 매핑 구축(정답 레퍼런스 역할)
    expected_kw_map, expected_strat_map, expected_bad_scope_map = build_expected(sc)
    expected_set = set(expected_kw_map.keys())

    # 1) 체크포인트 로드
    existing = None
    if RESUME and os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            print(f"기존 결과 로드: {len(existing)} rows")
        except Exception as e:
            print(f"기존 결과 로드 실패(무시): {e}")
            existing = None

    # 2) 작업 대상 키워드 집합(= expected_set) 기준으로 생성/업데이트
    records = []
    # 기존 결과를 dict로 맵핑(병합시 사용)
    prev_map: Dict[str, Dict] = {}
    if existing is not None and 'keyword' in existing.columns:
        for _, row in existing.iterrows():
            prev_map[str(row['keyword'])] = {
                'classification': row.get('classification','Pending'),
                'exists': row.get('exists',''),
                'strategy': row.get('strategy', row.get('strategy_used','')),
                'line_numbers': parse_line_list(row.get('line_numbers',''))
            }

    # 3) expected 기준으로 라인 검증 + 결과 구성
    processed = 0
    total = len(expected_set)
    print(f"총 {total}개의 npm 후보(기대값 기준)를 분석합니다.")

    for kw in sorted(expected_set):
        exp_lines = sorted(list(expected_kw_map.get(kw, set())))
        strat_name = expected_strat_map.get(kw, 'strat_comma')
        bad_scope = expected_bad_scope_map.get(kw, False)

        # 기존 라인과 병합(있다면)
        merged_lines = set(exp_lines)
        if kw in prev_map:
            merged_lines |= set(prev_map[kw].get('line_numbers', []))

        # 라인 검증: 각 라인에서 실제로 재현되는지 확인
        valid_lines = []
        # 전략 함수 선택
        strat_fn = None
        for key, fn in [
            ('strat_comma', strat_comma),
            ('strat_newline', strat_newline),
            ('strat_json', strat_json),
            ('strat_import', strat_import),
            ('strat_npm_install', strat_npm_install),
            ('strat_bullet', strat_bullet),
            ('strat_fallback', strat_fallback),
        ]:
            if key == strat_name:
                strat_fn = fn
                break
        if strat_fn is None:
            strat_fn = strat_comma

        for ln in sorted(merged_lines):
            sc_idx = ln - 2
            if sc_idx < 0 or sc_idx >= len(sc):  # 범위 밖 라인 제거
                continue
            rp = str(sc.loc[sc_idx, RESPONSE_COL]) if RESPONSE_COL in sc.columns else ""
            sp = str(sc.loc[sc_idx, SYSTEM_COL]) if SYSTEM_COL in sc.columns else ""
            # 우선 기록된 전략으로 검사, 실패시 system_prompt로 재선택 후 재시도
            cands = strat_fn(rp) if rp else []
            if not cands and strat_fn is not strat_fallback:
                cands = strat_fallback(rp)
            normed = set(n for n,_bad in (normalize_npm_name(x) for x in cands) if n)
            if kw not in normed:
                # 재선택
                fn2, _ = choose_strategy(sp)
                c2 = fn2(rp) if rp else []
                if not c2 and fn2 is not strat_fallback:
                    c2 = strat_fallback(rp)
                normed2 = set(n for n,_bad in (normalize_npm_name(x) for x in c2) if n)
                if kw in normed2:
                    valid_lines.append(ln)
            else:
                valid_lines.append(ln)

        # 존재성 확인 및 분류
        if kw in NODE_BUILTINS:
            classification = 'Built-in Module'
            exists = False
        elif kw in JS_KEYWORDS:
            classification = 'JS Keyword/Concept'
            exists = False
        elif bad_scope:
            # @scope 단독: 유지하되 Invalid로 고정
            classification = 'Unknown/Invalid'
            exists = False
        else:
            ex = exists_on_npm(kw)
            classification = 'NPM Package' if ex else 'Unknown/Invalid'
            exists = bool(ex)

        # 전략 이름(우선 expected의 것, 없으면 prev/map의 것)
        strategy_to_log = strat_name or (prev_map.get(kw, {}).get('strategy') if kw in prev_map else 'auto')

        records.append({
            'keyword': kw,
            'line_numbers': sorted(valid_lines),  # 검증된 라인만 담기
            'strategy': strategy_to_log,
            'classification': classification,
            'exists': exists
        })

        processed += 1
        print(f"[{processed}/{total}] {kw:40s} -> {classification} ({strategy_to_log}), lines={len(valid_lines)}")
        if RATE_SLEEP > 0: time.sleep(RATE_SLEEP)
        if processed % SAVE_INTERVAL == 0:
            save_csv(pd.DataFrame(records), OUTPUT_FILE)
            print(f"체크포인트 저장: {processed}/{total}")

    # 4) 최종 저장
    final_df = pd.DataFrame(records).sort_values('keyword').reset_index(drop=True)
    save_csv(final_df, OUTPUT_FILE)
    print(f"결과 저장: {OUTPUT_FILE}")

    # 5) 실행 후 SC↔FINAL 대조 리포트 자동 생성
    #    (과추출/누락/라인 재현 실패)
    # SC 재현(기대값)
    exp_map, exp_strat_map, _ = expected_kw_map, expected_strat_map, expected_bad_scope_map
    final_set = set(final_df['keyword'].astype(str))
    extra_in_final = sorted(list(final_set - set(exp_map.keys())))
    missing_in_final = sorted(list(set(exp_map.keys()) - final_set))

    # 라인 재현 실패(최종 결과의 각 line_numbers가 실제 그 줄에서 재현 가능한지 추가 점검)
    mismatch_rows = []
    strat_lookup = {
        'strat_comma': strat_comma,
        'strat_newline': strat_newline,
        'strat_json': strat_json,
        'strat_import': strat_import,
        'strat_npm_install': strat_npm_install,
        'strat_bullet': strat_bullet,
        'strat_fallback': strat_fallback,
    }
    for i, row in final_df.iterrows():
        kw = row['keyword']
        strat_name = str(row.get('strategy','strat_comma'))
        fn = strat_lookup.get(strat_name, strat_comma)
        for ln in row['line_numbers']:
            sc_idx = ln - 2
            if sc_idx < 0 or sc_idx >= len(sc):
                mismatch_rows.append((i, kw, ln, 'line_oob'))
                continue
            rp = str(sc.loc[sc_idx, RESPONSE_COL]) if RESPONSE_COL in sc.columns else ""
            sp = str(sc.loc[sc_idx, SYSTEM_COL]) if SYSTEM_COL in sc.columns else ""
            c1 = fn(rp) if rp else []
            if not c1 and fn is not strat_fallback:
                c1 = strat_fallback(rp)
            norm1 = set(n for n,_ in (normalize_npm_name(x) for x in c1) if n)
            if kw not in norm1:
                # 재선택 후 재검증
                fn2, _ = choose_strategy(sp)
                c2 = fn2(rp) if rp else []
                if not c2 and fn2 is not strat_fallback:
                    c2 = strat_fallback(rp)
                norm2 = set(n for n,_ in (normalize_npm_name(x) for x in c2) if n)
                if kw not in norm2:
                    mismatch_rows.append((i, kw, ln, 'not_found_in_line'))

    extra_df = pd.DataFrame({"keyword_extra_in_final": extra_in_final})
    missing_df = pd.DataFrame({"keyword_missing_in_final": missing_in_final})
    mismatch_df = pd.DataFrame(mismatch_rows, columns=["final_row_index","keyword","line_number","reason"])

    extra_path = "diff_extra_in_final.csv"
    missing_path = "diff_missing_in_final.csv"
    mismatch_path = "diff_line_mismatch.csv"
    extra_df.to_csv(extra_path, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")
    mismatch_df.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    print("\nDiff 리포트 생성 완료:")
    print(f"- 과추출(extra): {extra_path} ({len(extra_df)} rows)")
    print(f"- 누락(missing):  {missing_path} ({len(missing_df)} rows)")
    print(f"- 라인불일치:     {mismatch_path} ({len(mismatch_df)} rows)")

if __name__ == "__main__":
    main()
