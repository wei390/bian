import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import concurrent.futures
from threading import Lock, Semaphore
from tqdm import tqdm


class BinanceDataDownloader:
    def __init__(self, base_url="https://fapi.binance.com", data_dir="binance_data_0326", max_workers=5):
        self.base_url = base_url
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 初始化元数据
        self.metadata_file = os.path.join(data_dir, "metadata_0326.json")
        self.metadata = self._load_metadata()

        # 并发控制
        self.max_workers = max_workers
        self.write_lock = Lock()
        self.api_semaphore = Semaphore(15)  # 控制API并发数

        # 请求控制
        self.request_interval = 0.05  # 20 requests/second
        self.last_request_time = time.time()
        self.request_count = 0

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def get_all_symbols(self):
        url = f"{self.base_url}/fapi/v1/exchangeInfo"
        try:
            response = self._safe_api_call(url)
            if not response:
                return []
            return [s['symbol'] for s in response.json()['symbols']
                    if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        except Exception as e:
            print(f"获取交易对失败: {str(e)}")
            return []

    def _safe_api_call(self, url, params=None, retries=3):
        for attempt in range(retries):
            try:
                with self.api_semaphore:
                    # 控制请求频率
                    elapsed = time.time() - self.last_request_time
                    if elapsed < self.request_interval:
                        time.sleep(self.request_interval - elapsed)

                    response = requests.get(url, params=params, timeout=10)
                    self.last_request_time = time.time()
                    self.request_count += 1

                    # 处理限速
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 10))
                        print(f"触发限速，等待{retry_after}秒...")
                        time.sleep(retry_after)
                        continue

                    response.raise_for_status()
                    return response

            except Exception as e:
                if attempt == retries - 1:
                    print(f"API请求最终失败: {str(e)}")
                    return None
                wait = min(2 ** attempt, 10)
                print(f"请求失败，{wait}秒后重试...")
                time.sleep(wait)

        return None

    def download_kline_data(self, symbol, interval='15m', start_time=None, end_time=None):

        url = f"{self.base_url}/fapi/v1/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1500
        }

        # 时间范围验证
        now = datetime.now().timestamp() * 1000
        if start_time:
            start_ts = int(start_time.timestamp() * 1000)
            if start_ts > now:
                return []
            params['startTime'] = start_ts

        if end_time:
            end_ts = int(end_time.timestamp() * 1000)
            if end_ts > now:
                end_ts = int(now)
                # print(f"调整 {symbol} 结束时间为当前时间")
            params['endTime'] = end_ts

        data = []
        while True:
            response = self._safe_api_call(url, params)
            if not response:
                break

            batch = response.json()
            if not batch:
                break

            data.extend(batch)
            params['startTime'] = int(batch[-1][0]) + 1

            # 提前终止检查
            if params.get('endTime') and params['startTime'] > params['endTime']:
                break

        return data

    def _save_data(self, symbol, data):

        if not data:
            return

        try:

            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

            # 按月保存
            symbol_dir = os.path.join(self.data_dir, symbol)
            os.makedirs(symbol_dir, exist_ok=True)

            for month, group in df.groupby(df['open_time'].dt.to_period('M')):
                file_path = os.path.join(symbol_dir, f"{month}.csv")

                if os.path.exists(file_path):
                    existing = pd.read_csv(file_path, parse_dates=['open_time', 'close_time'])
                    combined = pd.concat([existing, group]).drop_duplicates('open_time').sort_values('open_time')
                    combined.to_csv(file_path, index=False)
                else:
                    group.to_csv(file_path, index=False)

            # 更新元数据
            last_ts = data[-1][0]
            with self.write_lock:
                self.metadata[symbol] = {
                    'last_updated': datetime.now().isoformat(),
                    'last_timestamp': last_ts
                }
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)

        except Exception as e:
            print(f"保存 {symbol} 数据失败: {str(e)}")

    def download_all_symbols(self, interval='15m', days=1):
        if days <= 0:
            raise ValueError("days参数必须大于0")

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        symbols = self.get_all_symbols()
        if not symbols:
            print("错误：未获取到交易对列表")
            return

        # print(f"获取到{len(symbols)}个交易对，示例：{symbols[:3]}...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._process_symbol, s, interval, start_time, end_time): s
                       for s in symbols}

            try:
                with tqdm(total=len(symbols), desc="总进度") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result(timeout=300)  # 5分钟超时
                            if result:
                                pbar.set_postfix_str(f"最后完成: {symbol}")
                        except concurrent.futures.TimeoutError:
                            print(f"\n{symbol} 处理超时，强制取消")
                            future.cancel()
                        except Exception as e:
                            print(f"\n{symbol} 处理异常: {str(e)}")
                        finally:
                            pbar.update(1)

            except KeyboardInterrupt:
                print("\n用户中断，取消剩余任务...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    def _process_symbol(self, symbol, interval, global_start, end_time):
        try:
            with self.write_lock:
                if symbol in self.metadata:
                    last_ts = self.metadata[symbol]['last_timestamp']
                    symbol_start = datetime.fromtimestamp(last_ts / 1000) + timedelta(milliseconds=1)
                else:
                    symbol_start = global_start

            if symbol_start > end_time:
                print(f"{symbol} 无需更新（本地数据已最新）")
                return (symbol, 0, "skip")

            data = self.download_kline_data(symbol, interval, symbol_start, end_time)

            if data:
                self._save_data(symbol, data)
                return (symbol, len(data), "success")
            return (symbol, 0, "no_data")

        except Exception as e:
            print(f"\n{symbol} 处理失败: {str(e)}")
            return (symbol, 0, "error")


if __name__ == "__main__":
    try:
        downloader = BinanceDataDownloader(max_workers=5)
        downloader.download_all_symbols(days=5)

    except Exception as e:
        print(f"程序异常终止: {str(e)}")
    finally:
        print("\n程序执行结束")