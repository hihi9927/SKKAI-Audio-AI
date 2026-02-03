#!/usr/bin/env python3
"""
Attention Log 분석 및 시각화 스크립트
Hallucination 감지를 위한 attention peak 패턴 분석
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_attention_log(csv_path):
    """CSV 파일에서 attention log 데이터 로드"""
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['avg_max_attention'] = float(row['avg_max_attention'])
            row['min_max_attention'] = float(row['min_max_attention'])
            row['max_max_attention'] = float(row['max_max_attention'])
            row['avg_peak_frame'] = float(row['avg_peak_frame'])
            row['token_count'] = int(row['token_count'])
            row['rewind_count'] = int(row['rewind_count'])
            row['is_monotonic'] = row['is_monotonic'] == 'True'
            row['per_token_data'] = json.loads(row['per_token_data'])
            data.append(row)
    return data


def classify_hallucination(row, avg_threshold=1.5, min_threshold=0.5):
    """
    Hallucination 여부 판정
    - avg_max_attention < 1.5 또는 min_max_attention < 0.5 → Hallucination
    """
    if row['avg_max_attention'] < avg_threshold or row['min_max_attention'] < min_threshold:
        return True
    return False


def analyze_and_plot(csv_path):
    """데이터 분석 및 그래프 생성"""
    data = load_attention_log(csv_path)

    if not data:
        print("데이터가 없습니다.")
        return

    # Hallucination 분류
    for row in data:
        row['is_hallucination'] = classify_hallucination(row)

    normal = [r for r in data if not r['is_hallucination']]
    hallucination = [r for r in data if r['is_hallucination']]

    print(f"\n=== 데이터 요약 ===")
    print(f"전체 세그먼트: {len(data)}")
    print(f"정상 문장: {len(normal)}")
    print(f"Hallucination 의심: {len(hallucination)}")

    # Figure 생성 (2x3 서브플롯)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Attention Peak 기반 Hallucination 분석', fontsize=14, fontweight='bold')

    # 1. avg_max_attention 분포 (히스토그램)
    ax1 = axes[0, 0]
    normal_avg = [r['avg_max_attention'] for r in normal]
    hall_avg = [r['avg_max_attention'] for r in hallucination]

    ax1.hist(normal_avg, bins=15, alpha=0.7, label=f'정상 (n={len(normal)})', color='green')
    ax1.hist(hall_avg, bins=15, alpha=0.7, label=f'Hallucination (n={len(hallucination)})', color='red')
    ax1.axvline(x=1.5, color='black', linestyle='--', label='Threshold (1.5)')
    ax1.set_xlabel('avg_max_attention')
    ax1.set_ylabel('빈도')
    ax1.set_title('평균 Max Attention 분포')
    ax1.legend()

    # 2. min_max_attention 분포
    ax2 = axes[0, 1]
    normal_min = [r['min_max_attention'] for r in normal]
    hall_min = [r['min_max_attention'] for r in hallucination]

    ax2.hist(normal_min, bins=15, alpha=0.7, label='정상', color='green')
    ax2.hist(hall_min, bins=15, alpha=0.7, label='Hallucination', color='red')
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax2.set_xlabel('min_max_attention')
    ax2.set_ylabel('빈도')
    ax2.set_title('최소 Max Attention 분포')
    ax2.legend()

    # 3. avg vs min scatter plot
    ax3 = axes[0, 2]
    for r in normal:
        ax3.scatter(r['avg_max_attention'], r['min_max_attention'],
                   c='green', alpha=0.6, s=50, marker='o')
    for r in hallucination:
        ax3.scatter(r['avg_max_attention'], r['min_max_attention'],
                   c='red', alpha=0.6, s=50, marker='x')

    # Threshold 영역 표시
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    ax3.fill_between([0, 1.5], [0, 0], [0.5, 0.5], alpha=0.1, color='red')
    ax3.set_xlabel('avg_max_attention')
    ax3.set_ylabel('min_max_attention')
    ax3.set_title('Avg vs Min Attention (Scatter)')
    ax3.legend(['정상', 'Hallucination'], loc='upper left')

    # 4. Rewind count 비교
    ax4 = axes[1, 0]
    normal_rewind = [r['rewind_count'] for r in normal]
    hall_rewind = [r['rewind_count'] for r in hallucination]

    categories = ['정상', 'Hallucination']
    avg_rewinds = [np.mean(normal_rewind) if normal_rewind else 0,
                   np.mean(hall_rewind) if hall_rewind else 0]
    colors = ['green', 'red']

    bars = ax4.bar(categories, avg_rewinds, color=colors, alpha=0.7)
    ax4.set_ylabel('평균 Rewind 횟수')
    ax4.set_title('Rewind Count 비교')
    for bar, val in zip(bars, avg_rewinds):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom')

    # 5. Monotonic 비율
    ax5 = axes[1, 1]
    normal_mono = sum(1 for r in normal if r['is_monotonic']) / len(normal) * 100 if normal else 0
    hall_mono = sum(1 for r in hallucination if r['is_monotonic']) / len(hallucination) * 100 if hallucination else 0

    bars = ax5.bar(categories, [normal_mono, hall_mono], color=colors, alpha=0.7)
    ax5.set_ylabel('Monotonic 비율 (%)')
    ax5.set_title('Attention Frame Monotonic 비율')
    ax5.set_ylim(0, 100)
    for bar, val in zip(bars, [normal_mono, hall_mono]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom')

    # 6. 토큰별 attention 분포 (boxplot)
    ax6 = axes[1, 2]
    normal_token_attn = []
    hall_token_attn = []

    for r in normal:
        for token in r['per_token_data']:
            normal_token_attn.append(token['max_attention'])
    for r in hallucination:
        for token in r['per_token_data']:
            hall_token_attn.append(token['max_attention'])

    bp = ax6.boxplot([normal_token_attn, hall_token_attn],
                     labels=['정상', 'Hallucination'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax6.set_ylabel('Token Max Attention')
    ax6.set_title('토큰별 Attention 분포 (Boxplot)')

    plt.tight_layout()

    # 저장
    output_path = csv_path.replace('.csv', '_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n그래프 저장: {output_path}")

    # 통계 출력
    print(f"\n=== 상세 통계 ===")
    print(f"\n[정상 문장]")
    if normal:
        print(f"  avg_max_attention: mean={np.mean(normal_avg):.3f}, std={np.std(normal_avg):.3f}")
        print(f"  min_max_attention: mean={np.mean(normal_min):.3f}, std={np.std(normal_min):.3f}")
        print(f"  monotonic 비율: {normal_mono:.1f}%")
        print(f"  avg rewind count: {np.mean(normal_rewind):.2f}")

    print(f"\n[Hallucination]")
    if hallucination:
        print(f"  avg_max_attention: mean={np.mean(hall_avg):.3f}, std={np.std(hall_avg):.3f}")
        print(f"  min_max_attention: mean={np.mean(hall_min):.3f}, std={np.std(hall_min):.3f}")
        print(f"  monotonic 비율: {hall_mono:.1f}%")
        print(f"  avg rewind count: {np.mean(hall_rewind):.2f}")

    # Hallucination 문장 목록
    print(f"\n=== Hallucination 의심 문장 목록 ===")
    for i, r in enumerate(hallucination, 1):
        sentence = r['sentence'][:50] + '...' if len(r['sentence']) > 50 else r['sentence']
        print(f"{i}. '{sentence}'")
        print(f"   avg_attn={r['avg_max_attention']:.3f}, min_attn={r['min_max_attention']:.3f}, "
              f"monotonic={r['is_monotonic']}, rewinds={r['rewind_count']}")

    plt.show()


if __name__ == '__main__':
    # 가장 최근 로그 파일 찾기
    log_dir = Path(__file__).parent
    log_files = sorted(log_dir.glob('attention_log_*.csv'), reverse=True)

    if not log_files:
        print("attention_log_*.csv 파일을 찾을 수 없습니다.")
        sys.exit(1)

    csv_path = str(log_files[0])
    print(f"분석 파일: {csv_path}")

    analyze_and_plot(csv_path)
