import os
import glob
import time
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


class BinanceDataStorage:
    def __init__(self, data_dir="binance_data", parquet_dir="binance_parquet",
                 cache_size=128, auto_init=True, incremental=True):

        self.data_dir = os.path.abspath(data_dir)
        self.parquet_dir = os.path.abspath(parquet_dir)
        self.cache_size = cache_size

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)

        if auto_init:
            need_init = not self._check_parquet_exists() or self._need_data_refresh()
            self.convert_data(incremental=incremental and need_init)

    def convert_data(self, incremental=True):

        print(f"\n{'增量' if incremental else '全量'}数据转换开始...")
        start_time = time.time()

        processed_count = 0
        for symbol_dir in glob.glob(os.path.join(self.data_dir, "*")):  # 遍历每个交易对目录
            if not os.path.isdir(symbol_dir):
                continue

            symbol = os.path.basename(symbol_dir)
            processed_count += self._process_symbol(symbol, incremental)

        # 更新_SUCCESS标记文件
        with open(os.path.join(self.parquet_dir, "_SUCCESS"), 'w') as f:
            f.write(datetime.now().isoformat())

        print(f"数据转换完成！处理 {processed_count} 个交易对，耗时 {time.time() - start_time:.2f} 秒")

    def _process_symbol(self, symbol, incremental):
        """处理单个交易对的数据转换（修复时间戳精度版本）"""
        symbol_dir = os.path.join(self.data_dir, symbol)
        parquet_dir = os.path.join(self.parquet_dir, symbol)

        # 获取需要处理的CSV文件（增量模式下过滤已处理文件）
        csv_files = glob.glob(os.path.join(symbol_dir, "*.csv"))
        if incremental and os.path.exists(parquet_dir):
            processed = self._get_processed_files(symbol)
            csv_files = [f for f in csv_files if os.path.basename(f) not in processed]

        if not csv_files:
            return 0  # 没有需要处理的文件

        # 读取CSV数据并强制类型转换
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(
                    csv_file,
                    parse_dates=['open_time', 'close_time'],
                    dtype={
                        'volume': 'float64',
                        'quote_volume': 'float64',
                        'count': 'float64',
                        'taker_buy_volume': 'float64',
                        'taker_buy_quote_volume': 'float64',
                        'ignore': 'float64'
                    }
                )
                dfs.append(df)
            except Exception as e:
                print(f"警告: 跳过损坏文件 {csv_file} - {str(e)}")
                continue

        # 合并数据并处理时间列
        new_data = pd.concat(dfs).sort_values('open_time').drop_duplicates(['open_time'])
        new_data['year'] = new_data['open_time'].dt.year
        new_data['month'] = new_data['open_time'].dt.month.astype(str)

        # 合并已有数据（增量模式）
        if incremental and os.path.exists(parquet_dir):
            try:
                existing = pd.read_parquet(parquet_dir)
                combined = pd.concat([existing, new_data]).sort_values('open_time')
                new_data = combined.drop_duplicates(['open_time'])
            except Exception as e:
                print(f"警告: 合并数据失败，将覆盖 {symbol} - {str(e)}")

        # 定义严格的PyArrow Schema（使用us精度）
        schema = pa.schema([
            ('open_time', pa.timestamp('us')),  # 改为us精度
            ('open', pa.float64()),
            ('high', pa.float64()),
            ('low', pa.float64()),
            ('close', pa.float64()),
            ('volume', pa.float64()),
            ('close_time', pa.timestamp('us')),  # 改为us精度
            ('quote_volume', pa.float64()),
            ('count', pa.float64()),
            ('taker_buy_volume', pa.float64()),
            ('taker_buy_quote_volume', pa.float64()),
            ('ignore', pa.float64()),
            ('month', pa.string()),
            ('year', pa.int32())
        ])

        # 写入Parquet（使用严格Schema）
        os.makedirs(parquet_dir, exist_ok=True)
        table = pa.Table.from_pandas(new_data, schema=schema, preserve_index=False)

        pq.write_to_dataset(
            table,
            root_path=parquet_dir,
            partition_cols=['year'],
            existing_data_behavior='delete_matching',
            version='2.6',
            coerce_timestamps='us'  # 改为us精度
        )

        # 记录已处理文件（增量模式）
        if incremental:
            self._update_processed_files(symbol, [os.path.basename(f) for f in csv_files])

        return 1  # 成功处理1个交易对

    def _check_parquet_exists(self):
        """检查Parquet数据是否已存在"""
        return (os.path.exists(os.path.join(self.parquet_dir, "_SUCCESS")) and
                len(glob.glob(os.path.join(self.parquet_dir, "*", "year=*", "*.parquet"))) > 0)

    def _need_data_refresh(self):
        """检查是否需要更新数据"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*", "*.csv"))
        if not csv_files:
            return False

        success_file = os.path.join(self.parquet_dir, "_SUCCESS")
        return (not os.path.exists(success_file) or (
                max(os.path.getmtime(f) for f in csv_files) > os.path.getmtime(success_file)))

    def _get_processed_files(self, symbol):
        """获取已处理的文件记录"""
        record_file = os.path.join(self.parquet_dir, symbol, "_processed.txt")
        if os.path.exists(record_file):
            with open(record_file, 'r') as f:
                return set(f.read().splitlines())
        return set()

    def _update_processed_files(self, symbol, files):
        """更新已处理文件记录"""
        record_file = os.path.join(self.parquet_dir, symbol, "_processed.txt")
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        with open(record_file, 'a') as f:
            f.write("\n".join(files) + "\n")

    def _get_parquet_files(self, symbol=None):
        """获取Parquet文件路径列表"""
        pattern = os.path.join(
            self.parquet_dir,
            symbol if symbol else "*",
            "year=*",
            "*.parquet"
        )
        return glob.glob(pattern, recursive=True)

    @lru_cache(maxsize=128)
    def _query_cache(self, symbol_hash, start_hash, end_hash):
        """带缓存的查询核心逻辑"""
        symbol = None if symbol_hash == "None" else symbol_hash
        start_time = datetime.fromisoformat(start_hash) if start_hash else None
        end_time = datetime.fromisoformat(end_hash) if end_hash else None

        parquet_files = self._get_parquet_files(symbol)
        # table = pq.read_table(parquet_files[0])  # 示例检查第一个文件
        # 检查所有文件的 "volume" 列是否包含小数
        # for f in parquet_files:
        #     df = pd.read_parquet(f)
        #     if any(df["volume"] % 1 != 0):
        #         print(f"文件 {f} 的 volume 列包含浮点值")
        # print("MMMMM")
        # print(table.schema)
        if not parquet_files:
            return pd.DataFrame()

        try:
            ddf = dd.read_parquet(
                parquet_files,
                engine="pyarrow",
                dtype={
                    "open_time": "datetime64[ms]",  # 确保时间列为 datetime
                    "close_time": "datetime64[ms]",
                    "volume": "float64",  # 其他数值列强制为 float
                    "quote_asset_volume": "float64"
                }
            )
            # ddf = dd.read_parquet(parquet_files)

            # 应用时间范围过滤
            if start_time:
                ddf = ddf[ddf['open_time'] >= pd.to_datetime(start_time)]
            if end_time:
                ddf = ddf[ddf['open_time'] <= pd.to_datetime(end_time)]

            return ddf.compute()  # 转换为 pandas DataFrame
        except Exception as e:
            print(f"查询错误: {str(e)}")
            return pd.DataFrame()

    def query_data(self, symbol=None, start_time=None, end_time=None):

        # 参数标准化
        symbol_hash = str(symbol) if symbol else "None"

        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        start_hash = start_time.isoformat() if start_time else ""

        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        end_hash = end_time.isoformat() if end_time else ""

        return self._query_cache(symbol_hash, start_hash, end_hash)

    def update_data(self):

        self.convert_data(incremental=True)
        self._query_cache.cache_clear()

    def clear_cache(self):
        """清除查询缓存"""
        self._query_cache.cache_clear()
        print("查询缓存已清除")

    def get_symbols(self):
        """获取所有交易对列表"""
        return sorted(
            d for d in os.listdir(self.parquet_dir)
            if os.path.isdir(os.path.join(self.parquet_dir, d)) and d != '_SUCCESS'
        )

    def get_data_stats(self):
        """获取数据统计信息（修正版）"""
        success_file = os.path.join(self.parquet_dir, "_SUCCESS")
        last_updated = None
        if os.path.exists(success_file):
            last_updated = datetime.fromtimestamp(os.path.getmtime(success_file))

        return {
            "data_dir": self.data_dir,
            "parquet_dir": self.parquet_dir,
            "last_updated": last_updated,
            "symbol_count": len(self.get_symbols()),
            "total_files": len(self._get_parquet_files()),
            "cache_info": self._query_cache.cache_info()
        }

    def benchmark_queries(self, test_symbol=None):
        """执行性能基准测试"""
        print("\n=== 性能基准测试 ===")

        # 测试1: 全量查询
        print("\n[测试1] 全量数据查询")
        start = time.perf_counter()
        df = self.query_data()
        elapsed = time.perf_counter() - start
        print(f"→ 行数: {len(df)} | 耗时: {elapsed:.4f}s")

        # 测试2: 单个交易对查询
        # symbol = test_symbol or self.get_symbols()[0]
        # print(f"\n[测试2] 单个交易对查询 ({symbol})")
        #
        # # 首次查询（缓存未命中）
        # start = time.perf_counter()
        # df = self.query_data(symbol=symbol)
        # elapsed = time.perf_counter() - start
        # print(f"→ 首次查询: {len(df)} 行 | 耗时: {elapsed:.4f}s")
        #
        # # 二次查询（缓存命中）
        # start = time.perf_counter()
        # df = self.query_data(symbol=symbol)
        # elapsed = time.perf_counter() - start
        # print(f"→ 缓存查询: {len(df)} 行 | 耗时: {elapsed:.4f}s")
        #
        # # 测试3: 时间范围查询
        # print(f"\n[测试3] 时间范围查询 ({symbol})")
        # start_time = datetime.now() - timedelta(days=7)
        # end_time = datetime.now()
        #
        # start = time.perf_counter()
        # df = self.query_data(
        #     symbol=symbol,
        #     start_time=start_time,
        #     end_time=end_time
        # )
        # elapsed = time.perf_counter() - start
        # print(f"→ 行数: {len(df)} | 耗时: {elapsed:.4f}s")
        # print(f"时间范围: {start_time} 到 {end_time}")


if __name__ == "__main__":
    # 示例用法
    print("=== 币安数据存储系统测试 ===")

    # 初始化存储系统
    storage = BinanceDataStorage(auto_init=True, incremental=True)

    # 显示系统状态
    stats = storage.get_data_stats()
    print("\n系统状态:")
    print(f"数据目录: {stats['data_dir']}")
    print(f"Parquet目录: {stats['parquet_dir']}")
    print(f"最后更新时间: {stats['last_updated'] or '从未更新'}")
    print(f"交易对数量: {stats['symbol_count']}")
    print(f"Parquet文件数: {stats['total_files']}")
    print("缓存状态:")
    print(f"  hits: {stats['cache_info'].hits}")
    print(f"  misses: {stats['cache_info'].misses}")
    print(f"  maxsize: {stats['cache_info'].maxsize}")
    print(f"  currsize: {stats['cache_info'].currsize}")

    # 执行性能测试
    if stats['symbol_count'] > 0:
        storage.benchmark_queries()
    else:
        print("\n警告: 没有可用的交易对数据")
