"""
학습 데이터 전처리 및 검증 유틸리티
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import argparse


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.stats = defaultdict(int)
    
    def scan_files(self) -> Dict[str, List[Path]]:
        """모든 코드 파일 스캔 (각 파일 타입은 선택적)"""
        files = {
            'html': [],
            'css': [],
            'js': [],
            'ts': []
        }
        
        if not self.data_dir.exists():
            print(f"경고: 디렉토리가 존재하지 않습니다: {self.data_dir}")
            return files
        
        # HTML 파일
        files['html'] = list(self.data_dir.rglob("*.html"))
        files['html'].extend(self.data_dir.rglob("*.htm"))
        
        # CSS 파일
        files['css'] = list(self.data_dir.rglob("*.css"))
        
        # JavaScript 파일
        files['js'] = list(self.data_dir.rglob("*.js"))
        files['js'].extend(self.data_dir.rglob("*.jsx"))
        
        # TypeScript 파일
        files['ts'] = list(self.data_dir.rglob("*.ts"))
        files['ts'].extend(self.data_dir.rglob("*.tsx"))
        
        return files
    
    def analyze_file(self, file_path: Path) -> Dict:
        """개별 파일 분석"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            chars = len(content)
            
            # 빈 줄 제외한 코드 라인
            code_lines = [line for line in lines if line.strip()]
            
            return {
                'path': str(file_path),
                'total_lines': len(lines),
                'code_lines': len(code_lines),
                'characters': chars,
                'size_kb': file_path.stat().st_size / 1024,
                'is_valid': chars > 0
            }
        except Exception as e:
            return {
                'path': str(file_path),
                'error': str(e),
                'is_valid': False
            }
    
    def generate_statistics(self) -> Dict:
        """데이터셋 통계 생성"""
        print(f"데이터 디렉토리 스캔 중: {self.data_dir}")
        files = self.scan_files()
        
        stats = {
            'total_files': 0,
            'by_type': {},
            'files': []
        }
        
        for file_type, file_list in files.items():
            print(f"\n{file_type.upper()} 파일 분석 중...")
            
            type_stats = {
                'count': len(file_list),
                'total_lines': 0,
                'total_chars': 0,
                'total_size_kb': 0,
                'valid_files': 0,
                'invalid_files': 0,
                'avg_lines': 0,
                'avg_chars': 0
            }
            
            for file_path in file_list:
                analysis = self.analyze_file(file_path)
                
                if analysis['is_valid']:
                    type_stats['valid_files'] += 1
                    type_stats['total_lines'] += analysis.get('total_lines', 0)
                    type_stats['total_chars'] += analysis.get('characters', 0)
                    type_stats['total_size_kb'] += analysis.get('size_kb', 0)
                    stats['files'].append({
                        'type': file_type,
                        **analysis
                    })
                else:
                    type_stats['invalid_files'] += 1
                    print(f"  경고: 유효하지 않은 파일 - {file_path}")
            
            # 평균 계산
            if type_stats['valid_files'] > 0:
                type_stats['avg_lines'] = type_stats['total_lines'] / type_stats['valid_files']
                type_stats['avg_chars'] = type_stats['total_chars'] / type_stats['valid_files']
            
            stats['by_type'][file_type] = type_stats
            stats['total_files'] += type_stats['count']
            
            print(f"  총 {type_stats['count']}개 파일")
            print(f"  유효: {type_stats['valid_files']}, 무효: {type_stats['invalid_files']}")
        
        return stats
    
    def print_statistics(self, stats: Dict):
        """통계 출력"""
        print("\n" + "=" * 70)
        print("데이터셋 통계")
        print("=" * 70)
        
        print(f"\n총 파일 수: {stats['total_files']}")
        
        for file_type, type_stats in stats['by_type'].items():
            print(f"\n{file_type.upper()} 파일:")
            print(f"  파일 수: {type_stats['count']}")
            print(f"  유효 파일: {type_stats['valid_files']}")
            print(f"  총 라인 수: {type_stats['total_lines']:,}")
            print(f"  총 문자 수: {type_stats['total_chars']:,}")
            print(f"  총 크기: {type_stats['total_size_kb']:.2f} KB")
            print(f"  평균 라인 수: {type_stats['avg_lines']:.1f}")
            print(f"  평균 문자 수: {type_stats['avg_chars']:.1f}")
        
        print("\n" + "=" * 70)
    
    def save_statistics(self, stats: Dict, output_file: str = "data_statistics.json"):
        """통계를 JSON 파일로 저장"""
        output_path = Path(output_file)
        
        # 파일 정보는 별도 저장 (용량 문제)
        files_info = stats.pop('files', [])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n통계 저장됨: {output_path}")
        
        # 파일 목록도 저장
        if files_info:
            files_path = output_path.parent / f"{output_path.stem}_files.json"
            with open(files_path, 'w', encoding='utf-8') as f:
                json.dump(files_info, f, indent=2, ensure_ascii=False)
            print(f"파일 목록 저장됨: {files_path}")
    
    def filter_large_files(
        self,
        max_lines: int = 1000,
        output_dir: str = None
    ) -> List[Path]:
        """너무 큰 파일 필터링"""
        files = self.scan_files()
        large_files = []
        
        for file_type, file_list in files.items():
            for file_path in file_list:
                analysis = self.analyze_file(file_path)
                
                if analysis.get('total_lines', 0) > max_lines:
                    large_files.append(file_path)
                    print(f"큰 파일 발견: {file_path} ({analysis['total_lines']} 라인)")
        
        if output_dir and large_files:
            output_path = Path(output_dir) / "large_files.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                for file_path in large_files:
                    f.write(f"{file_path}\n")
            print(f"\n큰 파일 목록 저장됨: {output_path}")
        
        return large_files
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """데이터셋 검증"""
        issues = []
        
        # 디렉토리 존재 확인
        if not self.data_dir.exists():
            issues.append(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
            return False, issues
        
        # 파일 스캔
        files = self.scan_files()
        total_files = sum(len(file_list) for file_list in files.values())
        
        if total_files == 0:
            issues.append("학습 가능한 파일이 없습니다.")
            return False, issues
        
        # 최소 파일 수 확인
        min_files = 100
        if total_files < min_files:
            issues.append(f"파일 수가 너무 적습니다. (최소 {min_files}개 권장, 현재 {total_files}개)")
        
        # 각 타입별 파일 확인 (경고만 표시, 필수 아님)
        for file_type, file_list in files.items():
            if len(file_list) == 0:
                issues.append(f"[경고] {file_type.upper()} 파일이 없습니다. (선택사항)")
        
        # 모든 파일 타입이 없으면 실패
        is_valid = total_files > 0
        
        return is_valid, issues


def main():
    parser = argparse.ArgumentParser(description="데이터 전처리 및 검증")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./training_data",
        help="학습 데이터 디렉토리"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_statistics.json",
        help="통계 저장 파일"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=1000,
        help="큰 파일 필터링 기준 (라인 수)"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="데이터셋 검증만 수행"
    )
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.data_dir)
    
    if args.validate:
        # 검증만 수행
        print("데이터셋 검증 중...")
        is_valid, issues = preprocessor.validate_dataset()
        
        if is_valid:
            print("\n✓ 데이터셋 검증 성공!")
        else:
            print("\n✗ 데이터셋 검증 실패:")
            for issue in issues:
                print(f"  - {issue}")
        
        if issues:
            print("\n경고:")
            for issue in issues:
                print(f"  - {issue}")
    else:
        # 전체 통계 생성
        stats = preprocessor.generate_statistics()
        preprocessor.print_statistics(stats)
        preprocessor.save_statistics(stats, args.output)
        
        # 큰 파일 필터링
        print(f"\n{args.max_lines} 라인 이상의 큰 파일 검색 중...")
        large_files = preprocessor.filter_large_files(
            max_lines=args.max_lines,
            output_dir=Path(args.data_dir).parent
        )
        
        if large_files:
            print(f"  발견된 큰 파일: {len(large_files)}개")
            print("  큰 파일은 학습 시 분할하거나 제외하는 것을 권장합니다.")


if __name__ == "__main__":
    main()
