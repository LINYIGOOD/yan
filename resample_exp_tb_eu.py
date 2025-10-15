#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
读取 HDF5（仅 /EXP_TB_EU 下变量），重采样到 30s：
- 模拟量(:ai/:ao) => 30s 均值
- 开关量(:bi/:bo/:SW) => 前向填充
输出单个 CSV（第一列 timestamp，后面每列是完整数据集路径）。
改进：
- 频率统一改为 "30s"（小写），消除 FutureWarning
- 先收集列后一次性 concat，避免 DataFrame 碎片化的 PerformanceWarning
"""

import glob
import argparse
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

# ====== 全局频率常量（小写）======
FREQ = "30s"

# ---------- 类型识别 ----------
ANALOG_SUFFIXES = (":ai", ":ao")         # 模拟量后缀
BINARY_SUFFIXES = (":bi", ":bo", ":sw")  # 开关量后缀（含 :SW）

def is_analog(var_name: str) -> bool:
    vn = var_name.lower()
    return any(vn.endswith(sfx) for sfx in ANALOG_SUFFIXES)

def is_binary(var_name: str) -> bool:
    vn = var_name.lower()
    return any(vn.endswith(sfx) for sfx in BINARY_SUFFIXES) or vn.endswith(":sw")

# ---------- HDF5 工具 ----------
def iter_variables_under_group(h5: h5py.File, group_path: str, debug: bool=False) -> List[str]:
    """
    返回 group_path 下所有**叶子数据集**（dataset）的绝对路径。
    注意：visititems 的 name 为相对该组的相对路径，需要补全。
    """
    if group_path not in h5:
        gp = group_path.strip("/")
        if gp not in h5:
            raise KeyError(f"HDF5 文件中不存在组 {group_path}")
        group_path = "/" + gp

    vars_paths: List[str] = []
    base = group_path.strip("/")   # "EXP_TB_EU"
    grp = h5[group_path]

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            full_path = f"/{base}/{name}"
            vars_paths.append(full_path)

    grp.visititems(visitor)

    if debug:
        print(f"[DEBUG] 在 {group_path} 下共发现 {len(vars_paths)} 个数据集（叶子）")
        for p in vars_paths[:10]:
            ds = h5[p]
            print(f"    - {p} shape={ds.shape} dtype={ds.dtype}")

    return vars_paths

def _as_numpy_array(x):
    """辅助：统一转 numpy 数组"""
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

def read_time_value_from_dataset(ds: h5py.Dataset, debug: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    从数据集中提取时间与数值两列，兼容：
      1) 复合 dtype，字段含 'time' / 'value'（或若干别名）
      2) 2D 普通数组，形如 [N,2]（第一列时间、第二列数值）
    返回：times(datetime64[ns]), values(float64)
    """
    arr = ds[()]

    # 情况 1：复合/结构化数组
    if hasattr(arr, "dtype") and arr.dtype.names:
        fields = {name.lower(): name for name in arr.dtype.names}
        time_keys = ["time", "timestamp", "ts", "date", "datetime"]
        value_keys = ["value", "val", "data", "y", "v"]

        t_key = next((fields[k] for k in time_keys if k in fields), None)
        v_key = next((fields[k] for k in value_keys if k in fields), None)
        if t_key is None or v_key is None:
            raise ValueError(f"结构化数据集找不到 time/value 字段: {ds.name}, 可用字段={arr.dtype.names}")

        times_raw = _as_numpy_array(arr[t_key])
        values_raw = _as_numpy_array(arr[v_key])

    else:
        # 情况 2：二维数组 [N,2]
        if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"数据集形状不符合 [N,2] 或结构化: {ds.name}, shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")
        times_raw = arr[:, 0]
        values_raw = arr[:, 1]

    # 处理 bytes -> str
    if getattr(times_raw, "dtype", None) is not None and (times_raw.dtype.kind in ("S", "O")):
        times_raw = np.array([t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else t for t in times_raw], dtype=object)

    # 转 datetime64[ns]
    times = pd.to_datetime(times_raw, errors="coerce").to_numpy(dtype="datetime64[ns]")

    # 数值：尽量转 float64
    if getattr(values_raw, "dtype", None) is not None and (values_raw.dtype.kind in ("S", "O")):
        values_raw = np.array([v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v for v in values_raw], dtype=object)
    values = pd.to_numeric(pd.Series(values_raw).astype(object), errors="coerce").to_numpy(dtype="float64")

    mask = ~pd.isna(times)
    times, values = times[mask], values[mask]

    if debug:
        print(f"[DEBUG] 读取 {ds.name}: 有效行数={len(times)}")

    return times, values

# ---------- 重采样 ----------
def build_global_index(series_meta: List[pd.DatetimeIndex], freq: str = FREQ) -> pd.DatetimeIndex:
    """用所有序列的时间范围构建统一的 30s 索引"""
    if not series_meta:
        raise RuntimeError("没有任何时间范围用于构建全局索引（可能没有成功读取到任何序列）。")
    min_ts = min(idx.min() for idx in series_meta if len(idx) > 0)
    max_ts = max(idx.max() for idx in series_meta if len(idx) > 0)
    start = pd.Timestamp(min_ts).floor(freq)
    end = pd.Timestamp(max_ts).ceil(freq)
    return pd.date_range(start=start, end=end, freq=freq)

def resample_one_series(s: pd.Series, var_name: str, global_index: pd.DatetimeIndex) -> pd.Series:
    """对单变量重采样并对齐到全局索引"""
    s = s.sort_index()
    if is_analog(var_name):
        out = s.resample(FREQ, label="left", closed="left").mean()
        out = out.reindex(global_index)
    elif is_binary(var_name):
        last_in_bin = s.resample(FREQ, label="left", closed="left").last()
        out = last_in_bin.reindex(global_index).ffill()
    else:
        out = s.resample(FREQ, label="left", closed="left").mean()
        out = out.reindex(global_index)
    return out

# ---------- 主流程 ----------
def process_files(input_glob: str, output_csv: str, group_name: str = "/EXP_TB_EU", debug: bool=False) -> None:
    series_meta: List[pd.DatetimeIndex] = []
    var_raw_cache: Dict[str, List[pd.Series]] = {}

    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"未匹配到任何文件: {input_glob}")
    if debug:
        print(f"[DEBUG] 将处理文件：{files}")

    for fp in files:
        with h5py.File(fp, "r") as h5:
            try:
                var_paths = iter_variables_under_group(h5, group_name, debug=debug)
            except KeyError:
                if debug:
                    print(f"[DEBUG] {fp} 不含组 {group_name}，跳过。")
                continue

            for vpath in var_paths:
                ds = h5[vpath]
                try:
                    times, values = read_time_value_from_dataset(ds, debug=debug)
                except Exception as e:
                    if debug:
                        print(f"[WARN] 跳过 {fp}{vpath}: {e}")
                    continue
                if len(times) == 0:
                    if debug:
                        print(f"[WARN] 跳过 {fp}{vpath}: 解析后无有效时间戳")
                    continue

                s = pd.Series(values, index=pd.to_datetime(times, utc=False))
                s = s[~s.index.duplicated(keep="last")]
                series_meta.append(pd.DatetimeIndex(s.index))

                col_name = vpath.strip("/")
                var_raw_cache.setdefault(col_name, []).append(s)

    if not var_raw_cache:
        raise RuntimeError(f"在以下文件的 {group_name} 下未找到可用变量: {files}")

    # 统一 30s 时间轴
    global_index = build_global_index(series_meta, freq=FREQ)

    # 一次性拼列，避免碎片化
    resampled_cols: List[pd.Series] = []
    for col_name, parts in sorted(var_raw_cache.items()):
        s_all = pd.concat(parts).sort_index()
        s_res = resample_one_series(s_all, col_name, global_index)
        resampled_cols.append(s_res.rename(col_name))

    # 如果变量很多，concat 会比逐列赋值快很多
    df_out = pd.concat(resampled_cols, axis=1)
    # 保证索引就是全局索引（理论上已对齐）
    df_out = df_out.reindex(global_index)

    # 写出 CSV：第一列为时间戳字符串（不含时区）
    df_out = df_out.reset_index(names="timestamp")
    df_out["timestamp"] = df_out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out.to_csv(output_csv, index=False)
    print(f"[OK] 已输出 {output_csv} ，形状={df_out.shape}")

def main():
    parser = argparse.ArgumentParser(description="将 HDF5 中 /EXP_TB_EU 下的变量重采样到 30s 并导出为 CSV。")
    parser.add_argument("--input", required=True, help="输入 HDF5 文件通配符或文件路径，例如 'C:/data/pv_*.hdf5'")
    parser.add_argument("--output", default="EXP_TB_EU_resampled_30s.csv", help="输出 CSV 路径")
    parser.add_argument("--group", default="/EXP_TB_EU", help="根组名称（默认 /EXP_TB_EU）")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    args = parser.parse_args()
    process_files(args.input, args.output, args.group, debug=args.debug)

if __name__ == "__main__":
    main()
