#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缺失值填补脚本（更新版）：
- 仅对模拟量(:ai/:ao)做前向填充（ffill）；开关量(:bi/:bo/:SW)按策略填充
- 内部缺口最大长度限制：--max-gap-points（默认 20）
- 端点外推可选：--extrapolate-ends none|both（默认 none）
  * 若 both，可再用 --max-end-gap-points 限制端点缺口长度（默认 20）
- 输出：填补后的 CSV + 报告 CSV（可选掩码 CSV）

依赖：pandas, numpy
"""

import argparse
from typing import List, Tuple, Literal, Optional
import numpy as np
import pandas as pd

# ---------- 列类型判定 ----------
def is_analog_col(name: str) -> bool:
    n = name.lower()
    return n.endswith(":ai") or n.endswith(":ao")

def is_binary_col(name: str) -> bool:
    n = name.lower()
    return n.endswith(":bi") or n.endswith(":bo") or n.endswith(":sw")

# ---------- 连续 NaN 段提取 ----------
def find_nan_runs(values: np.ndarray) -> List[Tuple[int, int]]:
    """
    返回所有连续 NaN 段的 [start_idx, end_idx]（闭区间，基于位置索引）
    """
    isn = np.isnan(values)
    runs: List[Tuple[int, int]] = []
    if not isn.any():
        return runs
    n = len(values)
    i = 0
    while i < n:
        if isn[i]:
            j = i
            while j + 1 < n and isn[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs

def classify_run(run: Tuple[int, int], total_len: int, has_left_value: bool, has_right_value: bool) -> Literal["internal","leading","trailing"]:
    """
    根据是否有左右已知值，将 NaN 段分类为 internal/leading/trailing
    """
    if has_left_value and has_right_value:
        return "internal"
    if (not has_left_value) and has_right_value:
        return "leading"
    return "trailing"

# ---------- 模拟量：前向填充（带缺口长度与端点策略） ----------
def impute_analog_series(
    s: pd.Series,
    max_gap_points: int = 200,                 # 模拟量内部缺口超过该长度则不填
    min_valid_points: int = 4,                # 兼容旧参数：前向填充实际只需 >=1 个有效点
    extrapolate_ends: Literal["none","both"] = "none",  # both 时允许端点外推
    max_end_gap_points: Optional[int] = 20,   # 端点外推时的最大缺口长度
) -> Tuple[pd.Series, pd.Series]:
    """
    对单列模拟量(:ai/:ao)进行缺失值填补（前向填充为主）：
    - internal 段（两侧都有已知值）：段长 ≤ max_gap_points 时，用 ffill 写回
    - trailing 段（只有左侧已知值）：若 extrapolate_ends == 'both' 且段长 ≤ max_end_gap_points，用 ffill 写回
    - leading  段（只有右侧已知值）：若 extrapolate_ends == 'both' 且段长 ≤ max_end_gap_points，用 bfill 写回
    其余位置保持 NaN。

    返回：
      filled_series（填补后的序列）, mask_filled（布尔序列，True 表示本次被填的位置）
    """
    x = s.values.astype(float)  # 包含 NaN
    n = len(x)
    out = s.copy()
    filled_mask = pd.Series(False, index=s.index)

    # 前向/后向填充至少需要 1 个有效点
    valid_mask = ~np.isnan(x)
    if valid_mask.sum() < 1:
        return out, filled_mask

    # 预先计算 ffill / bfill，后续按段选择性写回
    s_ff = s.ffill()
    s_bf = s.bfill()

    runs = find_nan_runs(x)
    for (a, b) in runs:
        length = b - a + 1

        has_left = (a - 1 >= 0) and valid_mask[a - 1]
        has_right = (b + 1 < n) and valid_mask[b + 1]
        kind = classify_run((a, b), n, has_left, has_right)

        allow_write = False
        vals = None  # 将要写回的序列切片

        if kind == "internal":
            allow_write = (length <= max_gap_points)
            if allow_write:
                vals = s_ff.iloc[a:b+1]

        elif kind == "trailing":
            if extrapolate_ends == "both":
                allow_write = (max_end_gap_points is None) or (length <= max_end_gap_points)
                if allow_write:
                    vals = s_ff.iloc[a:b+1]

        else:  # leading
            if extrapolate_ends == "both":
                allow_write = (max_end_gap_points is None) or (length <= max_end_gap_points)
                if allow_write:
                    vals = s_bf.iloc[a:b+1]

        if not allow_write or vals is None:
            continue

        # 写回该段值
        out.iloc[a:b+1] = vals.values

        # 掩码：标记“原来是 NaN 且这次被填上的位置”
        filled_mask.iloc[a:b+1] = s.iloc[a:b+1].isna() & vals.notna()

    return out, filled_mask

# ---------- 开关量：按策略填充 ----------
def impute_binary_series(
    s: pd.Series,
    mode: Literal["ffill","bfill","ffill_then_bfill"] = "ffill"
) -> Tuple[pd.Series, pd.Series]:
    """
    按策略对开关量填充：
    - ffill：只向前
    - bfill：只回填(用未来补过去)
    - ffill_then_bfill：先 ffill，再对仍为 NaN 的做 bfill
    返回：filled_series, mask_filled
    """
    orig_nan = s.isna()
    if mode == "ffill":
        y = s.ffill()
    elif mode == "bfill":
        y = s.bfill()
    elif mode == "ffill_then_bfill":
        y = s.ffill().bfill()
    else:
        y = s.copy()

    filled_mask = orig_nan & y.notna()
    return y, filled_mask

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description="缺失值填补：模拟量前向填充（可选端点外推），开关量可切换填充策略。")
    parser.add_argument("--input", required=True, help="输入 CSV（第一列 timestamp，后续为变量列）")
    parser.add_argument("--output", default="imputed.csv", help="输出 CSV（填补后）")
    parser.add_argument("--report", default=None, help="填补统计报告 CSV（默认：基于输出名自动生成）")
    parser.add_argument("--mask-output", default=None, help="可选：导出本次被填位置的掩码 CSV（0/1）")

    # 模拟量参数（保持兼容）
    parser.add_argument("--max-gap-points", type=int, default=200, help="内部缺口允许的最大连续 NaN 点数（默认 20）")
    parser.add_argument("--min-points-analog", type=int, default=4, help="（兼容项）前向填充仅需 >=1 个有效点")
    parser.add_argument("--extrapolate-ends", choices=["none","both"], default="none",
                        help="端点外推策略：none=不外推（默认），both=两端外推（leading 用 bfill，trailing 用 ffill）")
    parser.add_argument("--max-end-gap-points", type=int, default=20,
                        help="端点外推允许的最大连续 NaN 点数（仅在 extrapolate-ends=both 时应用，默认 20）")

    # 开关量参数
    parser.add_argument("--binary-fill", choices=["ffill","bfill","ffill_then_bfill"], default="ffill",
                        help="开关量缺失填充方式（默认 ffill）")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if df.shape[1] < 2:
        raise SystemExit("[ERROR] 输入 CSV 至少需要一列 timestamp 与一列数据列。")

    # 基本预处理：时间列在第 1 列（保持为字符串输出），其余列转 float
    time_col = df.columns[0]
    value_cols = df.columns.tolist()[1:]
    data = df.copy()
    for c in value_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # 统计与掩码容器
    report_rows: List[dict] = []
    mask_cols: List[pd.Series] = []

    # 按列处理
    for c in value_cols:
        s = data[c]
        if is_analog_col(c):
            y, m = impute_analog_series(
                s,
                max_gap_points=args.max_gap_points,
                min_valid_points=args.min_points_analog,
                extrapolate_ends=args.extrapolate_ends,
                max_end_gap_points=args.max_end_gap_points,
            )
            data[c] = y
            mask_cols.append(m.rename(c))
            report_rows.append({
                "column": c,
                "type": "analog",
                "method": "ffill",
                "extrapolate_ends": args.extrapolate_ends,
                "max_gap_points": args.max_gap_points,
                "max_end_gap_points": args.max_end_gap_points if args.extrapolate_ends=="both" else None,
                "orig_nan": int(s.isna().sum()),
                "filled": int(m.sum()),
                "remaining_nan": int(y.isna().sum()),
                "note": ""
            })
        elif is_binary_col(c):
            y, m = impute_binary_series(s, mode=args.binary_fill)
            data[c] = y
            mask_cols.append(m.rename(c))
            report_rows.append({
                "column": c,
                "type": "binary",
                "method": args.binary_fill,
                "extrapolate_ends": None,
                "max_gap_points": None,
                "max_end_gap_points": None,
                "orig_nan": int(s.isna().sum()),
                "filled": int(m.sum()),
                "remaining_nan": int(y.isna().sum()),
                "note": ""
            })
        else:
            # 既不是模拟量也不是开关量：不处理，只记录
            m = pd.Series(False, index=s.index, name=c)
            mask_cols.append(m)
            report_rows.append({
                "column": c,
                "type": "other",
                "method": "none",
                "extrapolate_ends": None,
                "max_gap_points": None,
                "max_end_gap_points": None,
                "orig_nan": int(s.isna().sum()),
                "filled": 0,
                "remaining_nan": int(s.isna().sum()),
                "note": "未识别类型，未处理"
            })

    # 写主结果
    data.to_csv(args.output, index=False)

    # 报告
    report_path = args.report if args.report else args.output.replace(".csv", "_impute_report.csv")
    pd.DataFrame(report_rows).sort_values(by=["type","filled"], ascending=[True,False]).to_csv(report_path, index=False)

    # 掩码（可选）
    if args.mask_output:
        mask_df = pd.concat(mask_cols, axis=1)
        mask_df.insert(0, time_col, data[time_col].values)
        # 将 True/False 转为 1/0
        for c in value_cols:
            mask_df[c] = mask_df[c].astype(int)
        mask_df.to_csv(args.mask_output, index=False)

    print(f"[OK] 已输出填补结果：{args.output}")
    print(f"[OK] 已输出统计报告：{report_path}")
    if args.mask_output:
        print(f"[OK] 已输出掩码文件：{args.mask_output}")

if __name__ == "__main__":
    main()


