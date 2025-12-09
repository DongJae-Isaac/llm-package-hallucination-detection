# RAG Verification Analysis
# RAG 적용 전후 환각 패키지 비교 및 npm registry 검증

import pandas as pd
import ast
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Configuration
RESULT_DIR = Path(r"d:\slopsquatting\result")
OUTPUT_DIR = Path(r"d:\slopsquatting\visualizations")

MODELS = ["gemma", "gpt_oss", "marin", "mistral", "qwen", "codellama7b"]

def extract_packages(response_npm):
    """response_npm_hallucinated에서 패키지 리스트 추출"""
    try:
        packages = ast.literal_eval(response_npm) if isinstance(response_npm, str) else response_npm
        if isinstance(packages, list):
            return [pkg.strip() for pkg in packages if pkg and isinstance(pkg, str) and pkg.strip()]
    except:
        pass
    return []

def extract_packages_from_rag_response(rag_response):
    """RAG 응답에서 추천된 패키지 추출"""
    import re
    
    if not isinstance(rag_response, str) or not rag_response.strip() or rag_response == 'nan':
        return []
    
    packages = set()
    
    # 패턴 1: npm install package-name 형태
    npm_install_pattern = r'npm\s+install\s+([a-zA-Z0-9@/_-]+)'
    matches = re.findall(npm_install_pattern, rag_response)
    packages.update(matches)
    
    # 패턴 2: require('package-name') 또는 import ... from 'package-name'
    require_pattern = r"(?:require|import|from)\s*\(\s*['\"]([a-zA-Z0-9@/_-]+)['\"]\s*\)"
    matches = re.findall(require_pattern, rag_response)
    packages.update(matches)
    
    # 패턴 3: "package-name" 또는 'package-name' (따옴표로 둘러싸인 것)
    quoted_pattern = r"['\"]([a-zA-Z0-9@/_-]{2,})['\"]\s*(?:package|library|module)"
    matches = re.findall(quoted_pattern, rag_response, re.IGNORECASE)
    packages.update(matches)
    
    # 패턴 4: 리스트나 배열 형태 ['pkg1', 'pkg2']
    try:
        # JSON이나 Python 리스트 형태 파싱 시도
        list_pattern = r'\[([^\]]+)\]'
        list_matches = re.findall(list_pattern, rag_response)
        for match in list_matches:
            items = re.findall(r"['\"]([a-zA-Z0-9@/_-]+)['\"]", match)
            packages.update(items)
    except:
        pass
    
    # 정리: 너무 짧거나 경로처럼 보이는 것 제외
    cleaned_packages = []
    for pkg in packages:
        pkg = pkg.strip()
        # 기본 필터링
        if not pkg or len(pkg) < 2:
            continue
        # 파일 경로 제외 (.js, .json 등)
        if pkg.endswith(('.js', '.json', '.ts', '.jsx', '.tsx')):
            continue
        # 상대 경로 제외
        if pkg.startswith('./') or pkg.startswith('../'):
            continue
        # URL 제외
        if pkg.startswith('http://') or pkg.startswith('https://'):
            continue
        
        cleaned_packages.append(pkg)
    
    return list(set(cleaned_packages))

def check_npm_registry(package_name, timeout=5):
    """npm registry에서 패키지 존재 여부 확인"""
    try:
        url = f"https://registry.npmjs.org/{package_name}"
        resp = requests.get(url, timeout=timeout)
        
        if resp.status_code == 200:
            data = resp.json()
            unpublished = data.get('time', {}).get('unpublished')
            deleted = data.get('_deleted', False)
            
            if unpublished or deleted:
                return False, resp.status_code
            else:
                return True, resp.status_code
        elif resp.status_code == 404:
            return False, resp.status_code
        else:
            return None, resp.status_code
    except:
        return None, -1

def analyze_model(model_name):
    """모델의 RAG 결과 검증"""
    
    # 원본 파일 (response_npm_hallucinated 포함)
    original_file = RESULT_DIR / f"result_paper_prompts_expanded_v2_out_{model_name}_final.csv"
    # RAG 파일 (result/rag 폴더 안에 있음)
    rag_file = RESULT_DIR / "rag" / f"result_rag_{model_name}.csv"
    
    if not original_file.exists():
        print(f"⚠️  {model_name}: 원본 파일 없음")
        return None
    
    if not rag_file.exists():
        print(f"⚠️  {model_name}: RAG 파일 없음")
        return None
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name.upper()}")
    print(f"{'='*80}")
    
    # 원본 파일 로드 (전체 83,540개)
    try:
        df_original = pd.read_csv(original_file, dtype=str, low_memory=False)
        print(f"Original total rows: {len(df_original):,}")
    except Exception as e:
        print(f"❌ Error loading original file: {e}")
        return None
    
    # RAG 파일 로드
    try:
        df_rag = pd.read_csv(rag_file, dtype=str, low_memory=False)
        print(f"RAG rows: {len(df_rag):,}")
    except Exception as e:
        print(f"❌ Error loading RAG file: {e}")
        return None
    
    # RAG를 딕셔너리로 변환 (original_index 기준)
    rag_dict = {}
    for idx, row in df_rag.iterrows():
        orig_idx = int(row.get('original_index', -1))
        if orig_idx >= 0:
            rag_dict[orig_idx] = row
    
    # 각 행을 처리하여 패키지별 데이터 수집
    all_packages_before = {}  # RAG 전 환각 패키지
    all_packages_after = {}   # RAG 후 환각 패키지
    
    # 행 수준 통계
    total_rows = len(df_original)
    rows_with_hallucination = 0  # 환각이 발생한 행 수 (원본)
    rows_with_rag = 0  # RAG가 적용된 행 수
    rows_after_rag = 0  # RAG 후 환각이 남은 행 수
    
    print("\n1단계: Extracting hallucinated packages from original file...")
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="행 처리"):
        # 원본 파일에서 환각 패키지 리스트 추출
        response_npm = str(row.get('response_npm_hallucinated', ''))
        
        if response_npm == 'nan' or not response_npm.strip():
            continue
        
        packages_before = extract_packages(response_npm)
        
        if not packages_before:
            continue
        
        # 이 행은 환각이 발생한 행
        rows_with_hallucination += 1
        
        # RAG 전 패키지 기록
        for pkg in packages_before:
            if pkg not in all_packages_before:
                all_packages_before[pkg] = {
                    'npm_exists': None,
                    'row_count': 0
                }
            all_packages_before[pkg]['row_count'] += 1
        
        # RAG 응답 찾기 및 분석
        if idx in rag_dict:
            rows_with_rag += 1
            rag_response = str(rag_dict[idx].get('rag_response', ''))
            
            # RAG 응답에서 추천된 패키지 추출
            packages_after = extract_packages_from_rag_response(rag_response)
            
            # RAG 후에도 환각 패키지가 남아있는지 확인
            # (원본 환각 패키지 중에서 RAG 후에도 추천된 것)
            remaining_hallucinations = set(packages_before) & set(packages_after)
            
            if remaining_hallucinations:
                rows_after_rag += 1
                
                # RAG 후에도 남은 환각 패키지 기록
                for pkg in remaining_hallucinations:
                    if pkg not in all_packages_after:
                        all_packages_after[pkg] = {
                            'npm_exists': None,
                            'row_count': 0
                        }
                    all_packages_after[pkg]['row_count'] += 1
    
    total_packages_before = len(all_packages_before)
    total_packages_after = len(all_packages_after)
    print(f"   Unique hallucinated packages before RAG: {total_packages_before:,}")
    print(f"   Unique hallucinated packages after RAG: {total_packages_after:,}")
    print(f"   Rows with hallucinations: {rows_with_hallucination:,} ({rows_with_hallucination/total_rows*100:.1f}%)")
    print(f"   Rows with RAG applied: {rows_with_rag:,}")
    print(f"   Rows with remaining hallucinations after RAG: {rows_after_rag:,}")
    
    # 2단계: npm registry 검증 (RAG 전 패키지)
    print(f"\n2단계: Verifying npm registry for packages before RAG (parallel processing)...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pkg = {
            executor.submit(check_npm_registry, pkg): pkg 
            for pkg in all_packages_before.keys()
        }
        
        for future in tqdm(as_completed(future_to_pkg), 
                          total=len(future_to_pkg),
                          desc=f"  {model_name.upper()} (Before)",
                          unit="pkg"):
            pkg = future_to_pkg[future]
            try:
                npm_exists, status_code = future.result()
                all_packages_before[pkg]['npm_exists'] = npm_exists
                all_packages_before[pkg]['status_code'] = status_code
            except Exception as e:
                all_packages_before[pkg]['npm_exists'] = None
                all_packages_before[pkg]['status_code'] = -1
    
    # 3단계: npm registry 검증 (RAG 후 패키지)
    if all_packages_after:
        print(f"\n3단계: Verifying npm registry for packages after RAG (parallel processing)...")
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_pkg = {
                executor.submit(check_npm_registry, pkg): pkg 
                for pkg in all_packages_after.keys()
            }
            
            for future in tqdm(as_completed(future_to_pkg), 
                              total=len(future_to_pkg),
                              desc=f"  {model_name.upper()} (After)",
                              unit="pkg"):
                pkg = future_to_pkg[future]
                try:
                    npm_exists, status_code = future.result()
                    all_packages_after[pkg]['npm_exists'] = npm_exists
                    all_packages_after[pkg]['status_code'] = status_code
                except Exception as e:
                    all_packages_after[pkg]['npm_exists'] = None
                    all_packages_after[pkg]['status_code'] = -1
    
    # 4단계: 통계 계산
    stats = {
        'model': model_name,
        'total_rows': total_rows,
        'rows_with_hallucination': rows_with_hallucination,
        'rows_with_rag': rows_with_rag,
        'rows_after_rag': rows_after_rag,
        'hallucination_rate_before': (rows_with_hallucination / total_rows * 100) if total_rows > 0 else 0,
        'hallucination_rate_after': (rows_after_rag / total_rows * 100) if total_rows > 0 else 0,
        'hallucination_reduction': 0.0,
        'total_packages_before': total_packages_before,
        'total_packages_after': total_packages_after,
        'npm_exists_before': 0,
        'npm_not_exists_before': 0,
        'npm_uncertain_before': 0,
        'npm_exists_after': 0,
        'npm_not_exists_after': 0,
        'npm_uncertain_after': 0,
        'npm_true_hallucination_rate_before': 0.0,
        'npm_true_hallucination_rate_after': 0.0
    }
    
    # 환각 감소율 계산
    stats['hallucination_reduction'] = stats['hallucination_rate_before'] - stats['hallucination_rate_after']
    
    # NPM 집계 (RAG 전)
    for pkg, data in all_packages_before.items():
        npm = data['npm_exists']
        
        if npm == True:
            stats['npm_exists_before'] += 1
        elif npm == False:
            stats['npm_not_exists_before'] += 1
        elif npm is None:
            stats['npm_uncertain_before'] += 1
    
    # NPM 집계 (RAG 후)
    for pkg, data in all_packages_after.items():
        npm = data['npm_exists']
        
        if npm == True:
            stats['npm_exists_after'] += 1
        elif npm == False:
            stats['npm_not_exists_after'] += 1
        elif npm is None:
            stats['npm_uncertain_after'] += 1
    
    # 실제 환각 비율 (RAG 전)
    npm_verifiable_before = stats['total_packages_before'] - stats['npm_uncertain_before']
    if npm_verifiable_before > 0:
        stats['npm_true_hallucination_rate_before'] = (stats['npm_not_exists_before'] / npm_verifiable_before) * 100
    
    # 실제 환각 비율 (RAG 후)
    npm_verifiable_after = stats['total_packages_after'] - stats['npm_uncertain_after']
    if npm_verifiable_after > 0:
        stats['npm_true_hallucination_rate_after'] = (stats['npm_not_exists_after'] / npm_verifiable_after) * 100
    
    # 출력
    print(f"\n{'='*80}")
    print(f"📊 Prompt-Level Statistics")
    print(f"{'='*80}")
    print(f"Total prompts: {stats['total_rows']:,}")
    print(f"\nBefore RAG:")
    print(f"  - Prompts with hallucinations: {stats['rows_with_hallucination']:,}")
    print(f"  - Hallucination rate: {stats['hallucination_rate_before']:.2f}%")
    
    print(f"\nRAG Applied:")
    print(f"  - Prompts with RAG: {stats['rows_with_rag']:,}")
    print(f"  - Prompts with remaining hallucinations: {stats['rows_after_rag']:,} (needs analysis)")
    print(f"  - Hallucination rate after RAG: {stats['hallucination_rate_after']:.2f}% (needs analysis)")
    print(f"  - Absolute reduction: +{stats['hallucination_reduction']:.2f}%p (needs analysis)")
    
    print(f"\n{'='*80}")
    print(f"📦 Package-Level Statistics")
    print(f"{'='*80}")
    print(f"Unique hallucinated packages before RAG: {stats['total_packages_before']:,}")
    
    print(f"\nnpm Registry Verification (packages before RAG):")
    if stats['total_packages_before'] > 0:
        print(f"  - Exists (false positive): {stats['npm_exists']:,} ({stats['npm_exists']/stats['total_packages_before']*100:.2f}%)")
        print(f"  - Does not exist (true hallucination): {stats['npm_not_exists']:,} ({stats['npm_true_hallucination_rate']:.2f}%)")
        print(f"  - Uncertain: {stats['npm_uncertain']:,}")
    
    # 상세 결과 저장
    output_file = OUTPUT_DIR / f"verify_rag_{model_name}.csv"
    results_df = pd.DataFrame([
        {
            'package': pkg,
            'npm_exists': data['npm_exists'],
            'row_count': data['row_count'],
            'status_code': data.get('status_code', -1)
        }
        for pkg, data in all_packages_before.items()
    ])
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nDetailed results saved: {output_file}")
    
    return stats

def create_summary_report(all_stats):
    """모델별 요약 리포트"""
    
    print(f"\n{'='*80}")
    print(f"📊 RAG Summary by Model")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<15} {'Before':<25} {'After':<25} {'Reduction':<30}")
    print("-" * 95)
    
    for stats in all_stats:
        if stats:
            before_count = stats['rows_with_hallucination']
            after_count = stats['rows_after_rag']
            reduced_count = before_count - after_count
            before_rate = stats['hallucination_rate_before']
            after_rate = stats['hallucination_rate_after']
            reduction_rate = stats['hallucination_reduction']
            
            print(f"{stats['model']:<15} "
                  f"{before_count:>6,} ({before_rate:>5.2f}%){'':<8} "
                  f"{after_count:>6,} ({after_rate:>5.2f}%){'':<8} "
                  f"-{reduced_count:>4,} (-{reduction_rate:>4.2f}%p)")
    
    print("\n" + "=" * 95)
    
    print("\n📝 Legend:")
    print("   - Before: Prompts with hallucinations before RAG (count and rate)")
    print("   - After: Prompts with hallucinations after RAG (count and rate)")
    print("   - Reduction: Number of prompts improved (count and percentage point decrease)")
    print("\n⚠️  Note: RAG response parsing logic needed for complete analysis")

def save_summary_csv(all_stats):
    """요약 통계 CSV 저장"""
    
    if not all_stats:
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "rag_accuracy_summary.csv"
    
    df = pd.DataFrame([s for s in all_stats if s])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSummary statistics saved: {output_file}")

def create_visualizations(all_stats):
    """시각화 생성: 환각 발생률 비교 막대 그래프 + Before-After 화살표 통합"""
    
    if not all_stats:
        return
    
    # 영어 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 데이터 준비
    models = [s['model'] for s in all_stats]
    before_rates = [s['hallucination_rate_before'] for s in all_stats]
    after_rates = [s['hallucination_rate_after'] for s in all_stats]
    reductions = [s['hallucination_reduction'] for s in all_stats]
    
    # Figure 생성
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # X축 위치 설정
    x = np.arange(len(models))
    width = 0.35
    
    # 막대 그래프
    bars1 = ax.bar(x - width/2, before_rates, width, label='Before RAG', 
                   color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.2, zorder=1)
    bars2 = ax.bar(x + width/2, after_rates, width, label='After RAG', 
                   color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.2, zorder=1)
    
    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(before_rates) * 0.02,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 화살표 오버레이
    for i, model in enumerate(models):
        before = before_rates[i]
        after = after_rates[i]
        
        x_pos = i
        
        # 연결점 강조
        ax.scatter([x_pos, x_pos], [before, after], s=200, 
                  color=['#FF6B6B', '#4ECDC4'], 
                  edgecolor='black', linewidth=2.5, zorder=5, alpha=0.9)
        
        # 화살표
        if before > after:
            ax.annotate('', xy=(x_pos, after), xytext=(x_pos, before),
                       arrowprops=dict(arrowstyle='->', lw=3.5, color='#2C3E50', 
                                     shrinkA=12, shrinkB=12, alpha=0.8), zorder=4)
            
            # 감소율 텍스트
            mid_y = (before + after) / 2
            ax.text(x_pos + 0.45, mid_y, f'-{reductions[i]:.2f}%p',
                   fontsize=10, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='#E74C3C', 
                            edgecolor='black', linewidth=1.5, alpha=0.9), zorder=6)
    
    # 축 설정
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Hallucination Rate Comparison Before and After RAG', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_ylim(0, max(before_rates) * 1.25)
    
    plt.tight_layout()
    
    # 저장
    output_file = OUTPUT_DIR / "rag_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    
    plt.close()
    
    print("✅ Visualization completed!")

def main():
    print("="*80)
    print("RAG Verification Analysis")
    print("Hallucinated packages before and after RAG vs npm registry")
    print("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    for model in MODELS:
        stats = analyze_model(model)
        if stats:
            all_stats.append(stats)
    
    if all_stats:
        create_summary_report(all_stats)
        save_summary_csv(all_stats)
        create_visualizations(all_stats)
    else:
        print("\nNo valid results found for analysis")
    
    print("\nVerification completed!")
    print("\n⚠️  Next step: Add rag_response parsing logic to complete RAG after-hallucination analysis")

if __name__ == "__main__":
    main()
