# Self-Refinement Verification Analysis V3
# 각 행의 환각 패키지를 npm registry로 검증하고 LLM 판단과 비교

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
SELF_REFINEMENT_DIR = RESULT_DIR / "self-refinement"
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

def extract_llm_judgment_for_package(response, package):
    """응답에서 특정 패키지에 대한 LLM 판단 추출"""
    if not isinstance(response, str) or not response.strip():
        return None
    
    # 패키지명 변형 (언더스코어 <-> 하이픈)
    variants = [package, package.replace('_', '-'), package.replace('-', '_')]
    
    # 응답에서 패키지가 언급된 줄 찾기
    lines = response.split('\n')
    for line in lines:
        line_upper = line.upper()
        
        # 이 줄에 패키지명이 있는지 확인
        has_package = False
        for variant in variants:
            if variant in line or variant.lower() in line.lower() or variant.upper() in line_upper:
                has_package = True
                break
        
        if not has_package:
            continue
        
        # 판단 추출 (순서 중요!)
        if any(phrase in line_upper for phrase in ['DOES NOT EXIST', 'DOES_NOT_EXIST', "DOESN'T EXIST", 'NOT EXIST']):
            return 'DOES_NOT_EXIST'
        elif 'EXISTS' in line_upper or 'EXIST' in line_upper:
            if 'NOT' not in line_upper and "N'T" not in line_upper:
                return 'EXISTS'
        elif 'UNCERTAIN' in line_upper or 'UNSURE' in line_upper:
            return 'UNCERTAIN'
    
    return None

def analyze_model(model_name):
    """모델의 self-refinement 결과 검증"""
    
    # 원본 파일 (response_npm_hallucinated 포함)
    original_file = RESULT_DIR / f"result_paper_prompts_expanded_v2_out_{model_name}_final.csv"
    # Self-refinement 파일
    refinement_file = SELF_REFINEMENT_DIR / f"result_self_refinement_{model_name}.csv"
    
    if not original_file.exists():
        print(f"⚠️  {model_name}: 원본 파일 없음")
        return None
    
    if not refinement_file.exists():
        print(f"⚠️  {model_name}: Self-refinement 파일 없음")
        return None
    
    print(f"\n{'='*80}")
    print(f"검증 중: {model_name.upper()}")
    print(f"{'='*80}")
    
    # 원본 파일 로드 (전체 83,540개)
    try:
        df_original = pd.read_csv(original_file, dtype=str, low_memory=False)
        print(f"원본 전체 행 수: {len(df_original):,}")
    except Exception as e:
        print(f"❌ 원본 파일 로딩 오류: {e}")
        return None
    
    # Self-refinement 파일 로드
    try:
        df_refinement = pd.read_csv(refinement_file, dtype=str, low_memory=False)
        print(f"Self-refinement 행 수: {len(df_refinement):,}")
    except Exception as e:
        print(f"❌ Self-refinement 파일 로딩 오류: {e}")
        return None
    
    # Self-refinement를 딕셔너리로 변환 (original_index 기준)
    refinement_dict = {}
    for idx, row in df_refinement.iterrows():
        orig_idx = int(row.get('original_index', -1))
        if orig_idx >= 0:
            refinement_dict[orig_idx] = row
    
    # 각 행을 처리하여 패키지별 데이터 수집
    all_packages_data = {}  # {package: {'npm_exists': bool, 'llm_judgments': [...]}}
    
    # 행 수준 통계
    total_rows = len(df_original)
    rows_with_hallucination = 0  # 환각이 발생한 행 수 (원본)
    rows_with_refinement = 0  # Self-refinement가 적용된 행 수
    rows_after_refinement = 0  # Self-refinement 후 환각이 남은 행 수
    
    print("\n1단계: 원본 파일에서 환각 패키지와 LLM 판단 추출 중...")
    for idx, row in tqdm(df_original.iterrows(), total=len(df_original), desc="행 처리"):
        # 원본 파일에서 환각 패키지 리스트 추출
        response_npm = str(row.get('response_npm_hallucinated', ''))
        
        if response_npm == 'nan' or not response_npm.strip():
            continue
        
        packages = extract_packages(response_npm)
        
        if not packages:
            continue
        
        # 이 행은 환각이 발생한 행
        rows_with_hallucination += 1
        
        # Self-refinement 응답 찾기
        refinement_response = ''
        if idx in refinement_dict:
            refinement_response = str(refinement_dict[idx].get('refinement_response', ''))
            rows_with_refinement += 1
        
        # Self-refinement 후 남은 패키지 (EXISTS로 판단된 것들)
        packages_after_refinement = []
        
        # 각 패키지에 대해 LLM 판단 추출
        for pkg in packages:
            if pkg not in all_packages_data:
                all_packages_data[pkg] = {
                    'npm_exists': None,  # 나중에 체크
                    'llm_judgments': [],
                    'row_count': 0  # 이 패키지가 등장한 행 수
                }
            
            all_packages_data[pkg]['row_count'] += 1
            
            # Self-refinement 응답에서 LLM 판단
            if refinement_response:
                judgment = extract_llm_judgment_for_package(refinement_response, pkg)
                if judgment:
                    all_packages_data[pkg]['llm_judgments'].append(judgment)
                    # EXISTS로 판단된 경우만 남김
                    if judgment == 'EXISTS':
                        packages_after_refinement.append(pkg)
            else:
                # Self-refinement 안된 경우 모두 유지
                packages_after_refinement.append(pkg)
        
        # Self-refinement 후에도 패키지가 남았으면 카운트
        if packages_after_refinement:
            rows_after_refinement += 1
    
    total_packages = len(all_packages_data)
    print(f"   고유 환각 패키지: {total_packages:,}개")
    print(f"   환각 발생 행: {rows_with_hallucination:,}개 ({rows_with_hallucination/total_rows*100:.1f}%)")
    print(f"   Self-refinement 적용 행: {rows_with_refinement:,}개")
    
    # 각 패키지의 LLM 판단 통합 (다수결)
    for pkg in all_packages_data:
        judgments = all_packages_data[pkg]['llm_judgments']
        if judgments:
            # 가장 많이 나온 판단 선택
            from collections import Counter
            most_common = Counter(judgments).most_common(1)[0][0]
            all_packages_data[pkg]['final_llm_judgment'] = most_common
        else:
            all_packages_data[pkg]['final_llm_judgment'] = 'NOT_JUDGED'
    
    # 2단계: npm registry 검증
    print(f"\n2단계: npm registry 검증 중 (병렬 처리)...")
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pkg = {
            executor.submit(check_npm_registry, pkg): pkg 
            for pkg in all_packages_data.keys()
        }
        
        for future in tqdm(as_completed(future_to_pkg), 
                          total=len(future_to_pkg),
                          desc=f"  {model_name.upper()}",
                          unit="pkg"):
            pkg = future_to_pkg[future]
            try:
                npm_exists, status_code = future.result()
                all_packages_data[pkg]['npm_exists'] = npm_exists
                all_packages_data[pkg]['status_code'] = status_code
            except Exception as e:
                all_packages_data[pkg]['npm_exists'] = None
                all_packages_data[pkg]['status_code'] = -1
    
    # 3단계: 통계 계산
    stats = {
        'model': model_name,
        'total_rows': total_rows,
        'rows_with_hallucination': rows_with_hallucination,
        'rows_with_refinement': rows_with_refinement,
        'rows_after_refinement': rows_after_refinement,
        'hallucination_rate_before': (rows_with_hallucination / total_rows * 100) if total_rows > 0 else 0,
        'hallucination_rate_after': (rows_after_refinement / total_rows * 100) if total_rows > 0 else 0,
        'hallucination_reduction': 0.0,  # 절대 감소율
        'total_packages': total_packages,
        'llm_judged': 0,
        'llm_not_judged': 0,
        'llm_exists': 0,
        'llm_not_exists': 0,
        'llm_uncertain': 0,
        'npm_exists': 0,
        'npm_not_exists': 0,
        'npm_uncertain': 0,
        'correct_exists': 0,
        'correct_not_exists': 0,
        'wrong_exists': 0,
        'wrong_not_exists': 0,
        'llm_accuracy': 0.0,
        'npm_true_hallucination_rate': 0.0
    }
    
    # 환각 감소율 계산 (절대값)
    stats['hallucination_reduction'] = stats['hallucination_rate_before'] - stats['hallucination_rate_after']
    
    for pkg, data in all_packages_data.items():
        llm = data['final_llm_judgment']
        npm = data['npm_exists']
        
        # LLM 판단 집계
        if llm == 'NOT_JUDGED':
            stats['llm_not_judged'] += 1
        else:
            stats['llm_judged'] += 1
            if llm == 'EXISTS':
                stats['llm_exists'] += 1
            elif llm == 'DOES_NOT_EXIST':
                stats['llm_not_exists'] += 1
            elif llm == 'UNCERTAIN':
                stats['llm_uncertain'] += 1
        
        # NPM 실제 집계
        if npm == True:
            stats['npm_exists'] += 1
        elif npm == False:
            stats['npm_not_exists'] += 1
        elif npm is None:
            stats['npm_uncertain'] += 1
        
        # 정확도 계산
        if llm not in ['NOT_JUDGED', 'UNCERTAIN'] and npm is not None:
            if llm == 'EXISTS' and npm == True:
                stats['correct_exists'] += 1
            elif llm == 'DOES_NOT_EXIST' and npm == False:
                stats['correct_not_exists'] += 1
            elif llm == 'EXISTS' and npm == False:
                stats['wrong_exists'] += 1
            elif llm == 'DOES_NOT_EXIST' and npm == True:
                stats['wrong_not_exists'] += 1
    
    # 정확도 계산
    if stats['llm_judged'] > 0:
        correct = stats['correct_exists'] + stats['correct_not_exists']
        judged_verifiable = stats['llm_judged'] - stats['llm_uncertain']
        if judged_verifiable > 0:
            stats['llm_accuracy'] = (correct / judged_verifiable) * 100
    
    # 실제 환각 비율
    npm_verifiable = stats['total_packages'] - stats['npm_uncertain']
    if npm_verifiable > 0:
        stats['npm_true_hallucination_rate'] = (stats['npm_not_exists'] / npm_verifiable) * 100
    
    # 출력
    print(f"\n{'='*80}")
    print(f"📊 프롬프트(행) 기준 통계")
    print(f"{'='*80}")
    print(f"전체 프롬프트: {stats['total_rows']:,}개")
    print(f"\n원본 (Self-Refinement 전):")
    print(f"  - 환각 발생 프롬프트: {stats['rows_with_hallucination']:,}개")
    print(f"  - 환각 발생률: {stats['hallucination_rate_before']:.2f}%")
    
    print(f"\nSelf-Refinement 적용:")
    print(f"  - 적용된 프롬프트: {stats['rows_with_refinement']:,}개")
    print(f"  - 적용 후 환각 남은 프롬프트: {stats['rows_after_refinement']:,}개")
    print(f"  - 적용 후 환각 발생률: {stats['hallucination_rate_after']:.2f}%")
    print(f"  - 절대 감소율: +{stats['hallucination_reduction']:.2f}%p")
    
    print(f"\n{'='*80}")
    print(f"📦 패키지 기준 통계")
    print(f"{'='*80}")
    print(f"고유 환각 패키지: {stats['total_packages']:,}개")
    
    print(f"\nSelf-Refinement LLM 판단 (패키지별):")
    if stats['total_packages'] > 0:
        print(f"  - 판단함: {stats['llm_judged']:,}개 ({stats['llm_judged']/stats['total_packages']*100:.2f}%)")
        print(f"  - 판단 안함: {stats['llm_not_judged']:,}개 ({stats['llm_not_judged']/stats['total_packages']*100:.2f}%)")
        if stats['llm_judged'] > 0:
            print(f"    -> EXISTS: {stats['llm_exists']:,}개")
            print(f"    -> DOES_NOT_EXIST: {stats['llm_not_exists']:,}개")
            print(f"    -> UNCERTAIN: {stats['llm_uncertain']:,}개")
    
    print(f"\nnpm Registry 실제 (패키지별):")
    if stats['total_packages'] > 0:
        print(f"  - 존재함 (오탐): {stats['npm_exists']:,}개 ({stats['npm_exists']/stats['total_packages']*100:.2f}%)")
        print(f"  - 존재하지 않음 (진짜 환각): {stats['npm_not_exists']:,}개 ({stats['npm_true_hallucination_rate']:.2f}%)")
        print(f"  - 불확실: {stats['npm_uncertain']:,}개")
    
    if stats['llm_judged'] > 0:
        print(f"\n{'='*80}")
        print(f"🎯 LLM 판단 정확도 (패키지별)")
        print(f"{'='*80}")
        print(f"  - 올바른 EXISTS 판단: {stats['correct_exists']:,}개")
        print(f"  - 올바른 DOES_NOT_EXIST 판단: {stats['correct_not_exists']:,}개")
        print(f"  - 잘못된 EXISTS 판단 (실제로는 없음): {stats['wrong_exists']:,}개")
        print(f"  - 잘못된 DOES_NOT_EXIST 판단 (실제로는 있음): {stats['wrong_not_exists']:,}개")
        print(f"\n  LLM 정확도: {stats['llm_accuracy']:.2f}%")
    
    # 상세 결과 저장
    output_file = OUTPUT_DIR / f"verify_{model_name}_v3.csv"
    results_df = pd.DataFrame([
        {
            'package': pkg,
            'npm_exists': data['npm_exists'],
            'llm_judgment': data['final_llm_judgment'],
            'llm_judgment_count': len(data['llm_judgments']),
            'status_code': data.get('status_code', -1)
        }
        for pkg, data in all_packages_data.items()
    ])
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n상세 결과 저장: {output_file}")
    
    return stats

def create_summary_report(all_stats):
    """모델별 요약 리포트"""
    
    print(f"\n{'='*80}")
    print(f"📊 Self-Refinement Summary by Model")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<15} {'Before':<25} {'After':<25} {'Reduction':<30}")
    print("-" * 95)
    
    for stats in all_stats:
        if stats:
            before_count = stats['rows_with_hallucination']
            after_count = stats['rows_after_refinement']
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
    print("   - Before: Prompts with hallucinations before self-refinement (count and rate)")
    print("   - After: Prompts with hallucinations after self-refinement (count and rate)")
    print("   - Reduction: Number of prompts improved (count and percentage point decrease)")

def save_summary_csv(all_stats):
    """요약 통계 CSV 저장"""
    
    if not all_stats:
        return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "self_refinement_accuracy_summary_v3.csv"
    
    df = pd.DataFrame([s for s in all_stats if s])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n요약 통계 저장: {output_file}")

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
    
    # ========================================
    # 막대 그래프 (배경)
    # ========================================
    bars1 = ax.bar(x - width/2, before_rates, width, label='Before', 
                   color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.2, zorder=1)
    bars2 = ax.bar(x + width/2, after_rates, width, label='After', 
                   color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.2, zorder=1)
    
    # 막대 위에 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(before_rates) * 0.02,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # ========================================
    # 화살표 오버레이 (전경)
    # ========================================
    for i, model in enumerate(models):
        before = before_rates[i]
        after = after_rates[i]
        
        # 막대 중심에서 화살표 그리기
        x_pos = i
        
        # 큰 점으로 연결점 강조
        ax.scatter([x_pos, x_pos], [before, after], s=200, 
                  color=['#FF6B6B', '#4ECDC4'], 
                  edgecolor='black', linewidth=2.5, zorder=5, alpha=0.9)
        
        # 화살표
        ax.annotate('', xy=(x_pos, after), xytext=(x_pos, before),
                   arrowprops=dict(arrowstyle='->', lw=3.5, color='#2C3E50', 
                                 shrinkA=12, shrinkB=12, alpha=0.8), zorder=4)
        
        # 감소율 텍스트 (화살표 옆)
        mid_y = (before + after) / 2
        ax.text(x_pos + 0.45, mid_y, f'-{reductions[i]:.2f}%p',
               fontsize=10, fontweight='bold', color='white',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='#E74C3C', 
                        edgecolor='black', linewidth=1.5, alpha=0.9), zorder=6)
    
    # 축 설정
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Hallucination Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Hallucination Rate Comparison Before and After Self-Refinement', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_ylim(0, max(before_rates) * 1.25)
    
    plt.tight_layout()
    
    # 저장
    output_file = OUTPUT_DIR / "self_refinement_comparison_v3.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n시각화 저장: {output_file}")
    
    plt.close()
    
    print("✅ 시각화 생성 완료!")

def main():
    print("="*80)
    print("Self-Refinement 검증 분석 V3")
    print("행별 환각 패키지 vs npm registry vs LLM 판단")
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
        print("\n분석할 유효한 결과를 찾을 수 없습니다")
    
    print("\n검증 완료!")

if __name__ == "__main__":
    main()
