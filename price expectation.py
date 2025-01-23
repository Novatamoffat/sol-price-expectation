import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
from pathlib import Path
import ta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class SOLPriceAnalyzer:
    def __init__(self):
        self.exchange = ccxt.okx({
            'enableRateLimit': True,
            'rateLimit': 100,  # 降低请求间隔
        })
        self.symbol = 'SOL/USDT'
        self.timeframe = '4h'  # 改用4小时k线，减少数据量
        self.cache_file = 'sol_price_cache.pkl'
        self.cache_validity = 3600  # 缓存有效期(秒)

    def _load_cache(self):
        """从缓存加载数据"""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < self.cache_validity:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        return None

    def _save_cache(self, data):
        """保存数据到缓存"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)

    def fetch_historical_data(self):
        """获取历史数据(带缓存)"""
        # 尝试从缓存加载
        cached_data = self._load_cache()
        if cached_data is not None:
            print("使用缓存数据")
            return cached_data

        try:
            print("从OKX获取新数据...")
            end_time = datetime.now()
            start_time = end_time - timedelta(days=90)
            
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                since=int(start_time.timestamp() * 1000),
                limit=1000
            )
            
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 添加技术指标
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            
            # 保存到缓存和CSV
            self._save_cache(df)
            df.to_csv(f'SOL_price_data_{datetime.now().strftime("%Y%m%d")}.csv')
            
            return df
            
        except Exception as e:
            print(f"获取数据时出错: {str(e)}")
            return None

    def prepare_features(self, df):
        """准备特征数据"""
        # 删除包含NaN的行
        df = df.dropna()
        
        # 创建特征
        X = df[['sma_20', 'rsi', 'macd', 'volume']].values
        y = df['close'].values
        
        # 分割训练集和测试集
        train_size = int(len(df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # 使用所有CPU核心
        )
        model.fit(X_train, y_train)
        return model

    def predict_future(self, model, df, days=7):
        """预测未来价格"""
        last_data = df.iloc[-1:]
        predictions = []
        current_features = last_data[['sma_20', 'rsi', 'macd', 'volume']].values
        
        for _ in range(days * 6):  # 4小时k线，一天6根
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            
            # 简单更新特征用于下一次预测
            current_features[0][0] = pred  # 更新SMA
            current_features[0][1] = min(max(current_features[0][1] * 0.95, 30), 70)  # RSI在30-70之间波动
            current_features[0][2] *= 0.95  # MACD衰减
            current_features[0][3] = current_features[0][3] * np.random.uniform(0.9, 1.1)  # 成交量随机波动
        
        return predictions

    def plot_results(self, df, predictions):
        """绘制结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        plt.plot(df.index[-30:], df['close'][-30:], label='历史价格')
        
        # 绘制预测数据
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=len(predictions) + 1,
            freq='4H'
        )[1:]
        plt.plot(future_dates, predictions, label='预测价格', linestyle='--')
        
        plt.title('SOL/USDT 价格预测 (优化版)')
        plt.xlabel('日期')
        plt.ylabel('价格 (USDT)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('SOL_price_prediction.png')
        plt.close()

def main():
    start_time = time.time()
    
    analyzer = SOLPriceAnalyzer()
    
    # 获取数据
    df = analyzer.fetch_historical_data()
    if df is None:
        return
    
    # 准备特征
    X_train, X_test, y_train, y_test = analyzer.prepare_features(df)
    
    # 训练模型
    print("训练模型中...")
    model = analyzer.train_model(X_train, y_train)
    
    # 预测
    predictions = analyzer.predict_future(model, df)
    
    # 绘制结果
    analyzer.plot_results(df, predictions)
    
    end_time = time.time()
    print(f"分析完成！用时: {end_time - start_time:.2f} 秒")
    print("请查看生成的CSV文件和预测图表。")

if __name__ == "__main__":
    main()












    import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
from pathlib import Path
import ta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

class SOLPriceAnalyzer:
    def __init__(self):
        self.base_url = "https://www.okx.com/api/v5"
        self.symbol = "SOL-USD"
        self.cache_file = 'sol_price_cache.pkl'
        self.cache_validity = 3600  # 缓存有效期(秒)

    def get_current_price(self):
        """获取当前价格"""
        url = f"{self.base_url}/market/ticker?instId={self.symbol}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            price = float(data['data'][0]['last'])
            print(f"{self.symbol} 当前价格: {price}")
            return price
        except Exception as e:
            print(f"获取当前价格时出错: {e}")
            return None

    def _load_cache(self):
        """从缓存加载数据"""
        cache_path = Path(self.cache_file)
        if cache_path.exists():
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age < self.cache_validity:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        return None

    def _save_cache(self, data):
        """保存数据到缓存"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)

    def fetch_historical_data(self):
        """获取历史数据"""
        # 尝试从缓存加载
        cached_data = self._load_cache()
        if cached_data is not None:
            print("使用缓存数据")
            return cached_data

        try:
            print("从OKX获取历史数据...")
            # 获取历史K线数据
            url = f"{self.base_url}/market/history-candles"
            params = {
                'instId': self.symbol,
                'bar': '4H',  # 4小时k线
                'limit': 500  # 获取500条数据
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()['data']
            
            # 转换数据格式
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'volCcy']
            )
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # 添加技术指标
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            
            # 保存到缓存和CSV
            self._save_cache(df)
            df.to_csv(f'SOL_price_data_{datetime.now().strftime("%Y%m%d")}.csv')
            
            return df
            
        except Exception as e:
            print(f"获取历史数据时出错: {str(e)}")
            return None

    def prepare_features(self, df):
        """准备特征数据"""
        df = df.dropna()
        
        X = df[['sma_20', 'rsi', 'macd', 'volume']].values
        y = df['close'].values
        
        train_size = int(len(df) * 0.8)
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        """训练随机森林模型"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

    def predict_future(self, model, df, days=7):
        """预测未来价格"""
        last_data = df.iloc[-1:]
        predictions = []
        current_features = last_data[['sma_20', 'rsi', 'macd', 'volume']].values
        
        for _ in range(days * 6):  # 4小时k线，一天6根
            pred = model.predict(current_features)[0]
            predictions.append(pred)
            
            # 更新特征
            current_features[0][0] = pred  # SMA
            current_features[0][1] = min(max(current_features[0][1] * 0.95, 30), 70)  # RSI
            current_features[0][2] *= 0.95  # MACD
            current_features[0][3] = current_features[0][3] * np.random.uniform(0.9, 1.1)  # Volume
        
        return predictions

    def plot_results(self, df, predictions, current_price=None):
        """绘制结果"""
        plt.figure(figsize=(12, 6))
        
        # 绘制历史数据
        plt.plot(df.index[-30:], df['close'][-30:], label='历史价格')
        
        # 绘制当前价格点
        if current_price:
            plt.scatter(df.index[-1], current_price, color='red', s=100, label='当前价格')
        
        # 绘制预测数据
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=len(predictions) + 1,
            freq='4H'
        )[1:]
        plt.plot(future_dates, predictions, label='预测价格', linestyle='--')
        
        plt.title('SOL-USD 价格预测')
        plt.xlabel('日期')
        plt.ylabel('价格 (USD)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig('SOL_price_prediction.png')
        plt.close()

def main():
    start_time = time.time()
    
    analyzer = SOLPriceAnalyzer()
    
    # 获取当前价格
    current_price = analyzer.get_current_price()
    
    # 获取历史数据
    df = analyzer.fetch_historical_data()
    if df is None:
        return
    
    # 准备特征
    X_train, X_test, y_train, y_test = analyzer.prepare_features(df)
    
    # 训练模型
    print("训练模型中...")
    model = analyzer.train_model(X_train, y_train)
    
    # 预测
    predictions = analyzer.predict_future(model, df)
    
    # 绘制结果
    analyzer.plot_results(df, predictions, current_price)
    
    end_time = time.time()
    print(f"分析完成！用时: {end_time - start_time:.2f} 秒")
    
    # 输出预测结果
    print("\n价格预测结果:")
    future_dates = pd.date_range(start=df.index[-1], periods=len(predictions)+1, freq='4H')[1:]
    for date, price in zip(future_dates[::6], predictions[::6]):  # 每天显示一个预测价格
        print(f"{date.strftime('%Y-%m-%d')}: {price:.2f} USD")

if __name__ == "__main__":
    main()