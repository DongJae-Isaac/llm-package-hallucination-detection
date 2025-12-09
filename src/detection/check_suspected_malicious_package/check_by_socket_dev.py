import csv
import requests
import json
import time
from typing import Dict, List, Set
from collections import defaultdict
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class NPMSecurityChecker:
    def __init__(self, socket_api_token: str):
        """
        NPM 패키지 보안 검사기 초기화
        
        Args:
            socket_api_token: Socket.dev API 토큰
        """
        self.socket_api_token = socket_api_token
        self.socket_base_url = "https://api.socket.dev/v0"
        self.headers = {
            "Authorization": f"Bearer {socket_api_token}",
            "Content-Type": "application/json"
        }
        
        # 재시도 전략이 포함된 세션 생성
        self.session = self._create_session()
        
    def _create_session(self):
        """재시도 로직이 포함된 requests 세션 생성"""
        session = requests.Session()
        
        # 재시도 전략 설정
        retry_strategy = Retry(
            total=3,  # 최대 3번 재시도
            backoff_factor=2,  # 2초, 4초, 8초로 증가
            status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 HTTP 상태 코드
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
        
    def extract_packages_from_csv(self, csv_file_path: str) -> List[str]:
        """
        CSV 파일에서 패키지명 추출 (순서 유지)
        
        Args:
            csv_file_path: CSV 파일 경로
            
        Returns:
            고유한 패키지명 리스트 (CSV 순서대로)
        """
        packages = []
        seen = set()
        
        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    package_name = row.get('package', '').strip()
                    if package_name and package_name != 'package' and package_name not in seen:
                        packages.append(package_name)
                        seen.add(package_name)
            
            print(f"✓ {len(packages)}개의 고유 패키지 발견 (CSV 순서 유지)")
            return packages
            
        except Exception as e:
            print(f"✗ CSV 파일 읽기 오류: {e}")
            return []
    
    def check_package_security(self, package_name: str, version: str = "latest", retry_count: int = 0) -> Dict:
        """
        Socket.dev API로 패키지 보안 점수 확인 (재시도 로직 포함)
        
        Args:
            package_name: 패키지명
            version: 패키지 버전 (기본값: latest)
            retry_count: 현재 재시도 횟수
            
        Returns:
            보안 점수 및 분석 결과
        """
        url = f"{self.socket_base_url}/npm/{package_name}/{version}/score"
        max_retries = 3
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)  # 30초로 증가
            
            if response.status_code == 404:
                return {
                    'status': 'not_found',
                    'package_name': package_name,
                    'error': 'Package not found in NPM registry'
                }
            
            # Rate limit 처리
            if response.status_code == 429:
                if retry_count < max_retries:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"  ⏳ Rate limit 도달. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                    return self.check_package_security(package_name, version, retry_count + 1)
                else:
                    return {
                        'status': 'rate_limited',
                        'package_name': package_name,
                        'error': 'Rate limit exceeded'
                    }
            
            response.raise_for_status()
            data = response.json()
            
            # Supply Chain Risk 분석
            supply_chain = data.get('supplyChainRisk', {})
            sc_score = supply_chain.get('score', 1.0)
            
            critical_issues = supply_chain.get('supplyChainRiskIssueCritical', 0)
            high_issues = supply_chain.get('supplyChainRiskIssueHigh', 0)
            mid_issues = supply_chain.get('supplyChainRiskIssueMid', 0)
            low_issues = supply_chain.get('supplyChainRiskIssueLow', 0)
            
            # 악성코드 판단
            is_malicious = self._evaluate_malicious(
                sc_score, critical_issues, high_issues, mid_issues
            )
            
            # 위험도 계산 (0-100%)
            supply_chain_risk_pct = (1 - sc_score) * 100
            
            return {
                'status': 'success',
                'package_name': package_name,
                'version': version,
                'supply_chain_score': sc_score,
                'supply_chain_risk_percentage': round(supply_chain_risk_pct, 2),
                'critical_issues': critical_issues,
                'high_issues': high_issues,
                'mid_issues': mid_issues,
                'low_issues': low_issues,
                'total_issues': critical_issues + high_issues + mid_issues + low_issues,
                'is_malicious': is_malicious,
                'risk_level': self._get_risk_level(sc_score, critical_issues, high_issues),
                'vulnerability_score': data.get('vulnerability', {}).get('score', 1.0),
                'quality_score': data.get('quality', {}).get('score', 1.0),
                'overall_score': data.get('depscore', 1.0)
            }
            
        except requests.exceptions.Timeout:
            if retry_count < max_retries:
                print(f"  ⏳ Timeout 발생. 재시도 중... ({retry_count + 1}/{max_retries})")
                time.sleep(5)  # 5초 대기 후 재시도
                return self.check_package_security(package_name, version, retry_count + 1)
            else:
                return {
                    'status': 'timeout',
                    'package_name': package_name,
                    'error': f'Request timeout after {max_retries} retries'
                }
        except requests.exceptions.RequestException as e:
            if retry_count < max_retries:
                print(f"  ⚠️ 요청 오류. 재시도 중... ({retry_count + 1}/{max_retries})")
                time.sleep(5)
                return self.check_package_security(package_name, version, retry_count + 1)
            else:
                return {
                    'status': 'error',
                    'package_name': package_name,
                    'error': str(e)
                }
    
    def check_typosquatting(self, package_name: str, retry_count: int = 0) -> Dict:
        """
        Socket.dev API로 typosquatting 확인 (재시도 로직 포함)
        
        Args:
            package_name: 패키지명
            retry_count: 현재 재시도 횟수
            
        Returns:
            Typosquatting 검사 결과
        """
        url = f"{self.socket_base_url}/npm/{package_name}/latest/issues"
        max_retries = 3
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=30)
            
            if response.status_code == 404:
                return {
                    'package_name': package_name,
                    'is_typosquatting': False,
                    'suggested_package': None,
                    'all_suggested_packages': [],
                    'typo_details': [],
                    'typo_severity': None,
                    'typo_count': 0
                }
            
            # Rate limit 처리
            if response.status_code == 429:
                if retry_count < max_retries:
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"  ⏳ Rate limit 도달. {wait_time}초 대기 중...")
                    time.sleep(wait_time)
                    return self.check_typosquatting(package_name, retry_count + 1)
            
            response.raise_for_status()
            data = response.json()
            
            # Typosquatting 관련 이슈 찾기
            typo_issues = []
            suggested_packages = []
            max_severity = None
            
            for issue in data:
                issue_type = issue.get('type', '').lower()
                issue_value = issue.get('value', {})
                
                # didYouMean 또는 gptDidYouMean 타입 확인
                if issue_type in ['didyoumean', 'gptdidyoumean']:
                    typo_issues.append(issue)
                    
                    # alternatePackage 추출
                    props = issue_value.get('props', {})
                    alternate_pkg = props.get('alternatePackage')
                    if alternate_pkg and alternate_pkg not in suggested_packages:
                        suggested_packages.append(alternate_pkg)
                    
                    # 심각도 확인 (가장 높은 심각도 저장)
                    severity = issue_value.get('severity', '').lower()
                    if severity == 'critical':
                        max_severity = 'critical'
                    elif severity == 'high' and max_severity != 'critical':
                        max_severity = 'high'
                    elif severity in ['middle', 'medium'] and max_severity not in ['critical', 'high']:
                        max_severity = 'medium'
                    elif severity == 'low' and max_severity is None:
                        max_severity = 'low'
            
            # 결과 반환
            is_typosquatting = len(typo_issues) > 0
            primary_suggestion = suggested_packages[0] if suggested_packages else None
            
            return {
                'package_name': package_name,
                'is_typosquatting': is_typosquatting,
                'suggested_package': primary_suggestion,
                'all_suggested_packages': suggested_packages,
                'typo_details': typo_issues,
                'typo_severity': max_severity,
                'typo_count': len(typo_issues)
            }
            
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if retry_count < max_retries:
                print(f"  ⚠️ 요청 오류. 재시도 중... ({retry_count + 1}/{max_retries})")
                time.sleep(3)
                return self.check_typosquatting(package_name, retry_count + 1)
            else:
                return {
                    'package_name': package_name,
                    'is_typosquatting': False,
                    'suggested_package': None,
                    'all_suggested_packages': [],
                    'typo_details': [],
                    'typo_severity': None,
                    'typo_count': 0,
                    'error': f'Failed to check typosquatting after retries: {str(e)}'
                }
    
    def _evaluate_malicious(self, sc_score: float, critical: int, high: int, mid: int) -> bool:
        """악성코드 여부 판단"""
        if critical >= 1:
            return True
        if sc_score <= 0.4:
            return True
        if high >= 3:
            return True
        if (1 - sc_score) >= 0.7 and high >= 1:
            return True
        return False
    
    def _get_risk_level(self, sc_score: float, critical: int, high: int) -> str:
        """위험 수준 분류"""
        if critical >= 1 or sc_score <= 0.3:
            return "CRITICAL"
        elif high >= 2 or sc_score <= 0.5:
            return "HIGH"
        elif sc_score <= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _load_processed_packages(self, output_file: str) -> List[str]:
        """이미 처리된 패키지 목록 로드 (순서 유지)"""
        processed = []
        seen = set()
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    package_name = row.get('package_name', '').strip()
                    if package_name and package_name not in seen:
                        processed.append(package_name)
                        seen.add(package_name)
            
            if processed:
                print(f"✓ 이전 진행 상황 발견: {len(processed)}개 패키지 이미 처리됨")
        except FileNotFoundError:
            print(f"✓ 새로운 검사 시작")
        
        return processed
    
    def _append_to_csv(self, result: Dict, output_file: str, write_header: bool = False):
        """결과를 CSV 파일에 추가 저장"""
        fieldnames = [
            'package_name', 'version', 'status',
            'supply_chain_risk_percentage', 'risk_level',
            'is_malicious', 'is_typosquatting', 'suggested_package', 'all_suggested_packages',
            'typo_severity', 'typo_count',
            'critical_issues', 'high_issues', 'mid_issues', 'low_issues', 'total_issues',
            'supply_chain_score', 'vulnerability_score', 'quality_score', 'overall_score'
        ]
        
        mode = 'w' if write_header else 'a'
        
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            
            # all_suggested_packages를 문자열로 변환
            result_copy = result.copy()
            if 'all_suggested_packages' in result_copy and isinstance(result_copy['all_suggested_packages'], list):
                result_copy['all_suggested_packages'] = ', '.join(result_copy['all_suggested_packages'])
            
            writer.writerow(result_copy)
    
    def bulk_check_packages(self, packages: List[str], output_file: str = "security_results.csv", 
                           checkpoint_interval: int = 10, delay_between_requests: float = 1.5):
        """
        여러 패키지를 일괄 검사 (개선된 버전, 순서 유지)
        
        Args:
            packages: 검사할 패키지 리스트 (순서 유지)
            output_file: 결과 저장 파일명
            checkpoint_interval: 중간 저장 간격
            delay_between_requests: 요청 간 대기 시간 (초) - 기본 1.5초로 증가
        """
        processed_packages = self._load_processed_packages(output_file)
        processed_set = set(processed_packages)
        
        # 이미 처리된 패키지를 제외하고 순서 유지
        remaining_packages = [pkg for pkg in packages if pkg not in processed_set]
        
        if not remaining_packages:
            print("\n✓ 모든 패키지가 이미 처리되었습니다!")
            results = self._load_all_results(output_file)
            self._print_statistics(results)
            return results
        
        results = []
        total = len(packages)
        processed_count = len(processed_packages)
        remaining_count = len(remaining_packages)
        
        print(f"\n{'='*70}")
        print(f"총 {total}개 패키지 보안 검사")
        print(f"이미 처리됨: {processed_count}개")
        print(f"남은 패키지: {remaining_count}개")
        print(f"요청 간 대기 시간: {delay_between_requests}초")
        print(f"{'='*70}\n")
        
        write_header = processed_count == 0
        
        for idx, package_name in enumerate(remaining_packages, 1):
            current_total = processed_count + idx
            print(f"[{current_total}/{total}] 검사 중: {package_name}")
            
            # 보안 점수 확인
            security_result = self.check_package_security(package_name)
            
            # API 속도 제한 고려 (증가된 대기 시간)
            time.sleep(delay_between_requests)
            
            # Typosquatting 확인
            typo_result = self.check_typosquatting(package_name)
            
            # 결과 병합
            if security_result['status'] == 'success':
                combined_result = {
                    **security_result,
                    **typo_result
                }
                results.append(combined_result)
                
                if combined_result.get('is_malicious') or combined_result.get('is_typosquatting'):
                    self._print_alert(combined_result)
            else:
                combined_result = {
                    'package_name': package_name,
                    'status': security_result['status'],
                    'error': security_result.get('error', 'Unknown error'),
                    'is_typosquatting': typo_result.get('is_typosquatting', False),
                    'suggested_package': typo_result.get('suggested_package'),
                    'all_suggested_packages': typo_result.get('all_suggested_packages', []),
                    'typo_severity': typo_result.get('typo_severity'),
                    'typo_count': typo_result.get('typo_count', 0)
                }
                results.append(combined_result)
                print(f"  ⚠️ 상태: {security_result['status']} - {security_result.get('error', '')}")
                
                # 보안 점수는 실패했지만 typosquatting은 발견된 경우
                if combined_result.get('is_typosquatting'):
                    self._print_alert(combined_result)
            
            # 중간 저장
            if idx % checkpoint_interval == 0 or idx == remaining_count:
                start_idx = max(0, len(results) - checkpoint_interval)
                for result in results[start_idx:]:
                    self._append_to_csv(result, output_file, write_header)
                    write_header = False
                
                print(f"💾 중간 저장 완료: {current_total}/{total} ({current_total/total*100:.1f}%)")
            
            time.sleep(delay_between_requests)  # 추가 대기
        
        all_results = self._load_all_results(output_file)
        self._print_statistics(all_results)
        
        return all_results
    
    def _load_all_results(self, output_file: str) -> List[Dict]:
        """저장된 모든 결과 로드"""
        results = []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 숫자 필드 변환
                    if row.get('supply_chain_risk_percentage'):
                        try:
                            row['supply_chain_risk_percentage'] = float(row['supply_chain_risk_percentage'])
                        except ValueError:
                            row['supply_chain_risk_percentage'] = 0.0
                    
                    for field in ['critical_issues', 'high_issues', 'mid_issues', 'low_issues', 'typo_count']:
                        if row.get(field):
                            try:
                                row[field] = int(row[field])
                            except ValueError:
                                row[field] = 0
                    
                    # 불린 필드 변환
                    for field in ['is_malicious', 'is_typosquatting']:
                        if row.get(field):
                            row[field] = row[field].lower() == 'true'
                    
                    # 리스트 필드 변환
                    if row.get('all_suggested_packages'):
                        row['all_suggested_packages'] = [pkg.strip() for pkg in row['all_suggested_packages'].split(',') if pkg.strip()]
                    
                    results.append(row)
        except FileNotFoundError:
            pass
        
        return results
    
    def _print_alert(self, result: Dict):
        """위험한 패키지 경고 출력"""
        print(f"\n{'⚠️ '*20}")
        print(f"위험 패키지 발견: {result['package_name']}")
        
        if result.get('is_malicious'):
            print(f"  🔴 악성코드 의심: YES")
            risk_pct = result.get('supply_chain_risk_percentage', 0)
            print(f"  📊 Supply Chain Risk: {risk_pct:.1f}%")
            print(f"  🚨 위험 수준: {result.get('risk_level', 'UNKNOWN')}")
            print(f"  ⚠️  Critical 이슈: {result.get('critical_issues', 0)}")
            print(f"  ⚠️  High 이슈: {result.get('high_issues', 0)}")
        
        if result.get('is_typosquatting'):
            print(f"  🔍 Typosquatting 발견!")
            typo_severity = result.get('typo_severity', 'Unknown')
            print(f"  📈 심각도: {typo_severity.upper() if typo_severity else 'UNKNOWN'}")
            print(f"  🔢 Typo 이슈 수: {result.get('typo_count', 0)}")
            
            if result.get('suggested_package'):
                print(f"  💡 주요 추천 패키지: {result['suggested_package']}")
            
            all_suggestions = result.get('all_suggested_packages', [])
            if isinstance(all_suggestions, list) and len(all_suggestions) > 1:
                print(f"  📋 모든 추천 패키지: {', '.join(all_suggestions)}")
            elif isinstance(all_suggestions, str) and ',' in all_suggestions:
                print(f"  📋 모든 추천 패키지: {all_suggestions}")
        
        print(f"{'⚠️ '*20}\n")
    
    def _print_statistics(self, results: List[Dict]):
        """검사 통계 출력"""
        if not results:
            print("\n통계를 출력할 결과가 없습니다.")
            return
        
        successful = [r for r in results if r.get('status') == 'success']
        malicious = [r for r in successful if r.get('is_malicious')]
        typosquatting = [r for r in results if r.get('is_typosquatting')]  # 전체 결과에서 검색
        errors = [r for r in results if r.get('status') in ['timeout', 'error', 'rate_limited']]
        
        print(f"\n{'='*70}")
        print("검사 결과 요약")
        print(f"{'='*70}")
        print(f"총 검사 패키지: {len(results)}")
        print(f"성공적으로 검사됨: {len(successful)}")
        print(f"오류 발생: {len(errors)}")
        
        if len(results) > 0:
            print(f"악성코드 의심 패키지: {len(malicious)} ({len(malicious)/len(results)*100:.1f}%)")
            print(f"Typosquatting 패키지: {len(typosquatting)} ({len(typosquatting)/len(results)*100:.1f}%)")
        
        if malicious:
            print(f"\n🔴 악성코드 의심 패키지 목록:")
            for pkg in malicious[:10]:
                risk_pct = pkg.get('supply_chain_risk_percentage', 0)
                risk_level = pkg.get('risk_level', 'UNKNOWN')
                print(f"  - {pkg['package_name']} (위험도: {risk_pct:.1f}%, 수준: {risk_level})")
            if len(malicious) > 10:
                print(f"  ... 외 {len(malicious) - 10}개")
        
        if typosquatting:
            print(f"\n🔍 Typosquatting 패키지 목록:")
            for pkg in typosquatting[:10]:
                suggestion = pkg.get('suggested_package', 'N/A')
                severity = pkg.get('typo_severity', 'Unknown')
                print(f"  - {pkg['package_name']} → {suggestion} (심각도: {severity.upper() if severity else 'UNKNOWN'})")
            if len(typosquatting) > 10:
                print(f"  ... 외 {len(typosquatting) - 10}개")
        
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # ==============================================================
    # !api, csv_file명 수정!
    SOCKET_API_TOKEN = ""
    CSV_FILE = "stats_paper_prompts_expanded_v2_out_marin_final.csv"
    OUTPUT_FILE = "npm_security_check_results_marin_socket_dev.csv"
    # ==============================================================
    CHECKPOINT_INTERVAL = 10
    
    checker = NPMSecurityChecker(SOCKET_API_TOKEN)
    
    print("CSV 파일에서 패키지 추출 중...")
    packages = checker.extract_packages_from_csv(CSV_FILE)
    
    if packages:
        # delay_between_requests를 2.0초로 설정하여 rate limit 방지
        results = checker.bulk_check_packages(
            packages, 
            output_file=OUTPUT_FILE,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            delay_between_requests=2.0  # API 요청 간격을 2초로 증가
        )
        
        print("\n✅ 검사 완료!")
        print(f"자세한 결과는 '{OUTPUT_FILE}' 파일을 확인하세요.")
    else:
        print("검사할 패키지가 없습니다.")
