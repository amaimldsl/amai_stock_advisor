import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import finnhub
from fredapi import Fred

class EnhancedStockAnalysis:
    def __init__(self, api_keys):
        """Initialize with API keys for external data sources"""
        self.finnhub_client = finnhub.Client(api_key=api_keys['finnhub'])
        self.fred = Fred(api_key=api_keys['fred'])
        self.scaler = MinMaxScaler()
        
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators using pandas"""
        df = data.copy()
        
        # Print the column structure for debugging
        print(f"\nInitial columns structure:")
        print(f"Column names: {df.columns.values}")
        print(f"Column type: {type(df.columns)}")
        
        # If we have a MultiIndex in columns, get the first level
        if isinstance(df.columns, pd.MultiIndex):
            # Use the first level of the MultiIndex
            print("Converting MultiIndex columns...")
            close_price = df['Close']
            high_price = df['High']
            low_price = df['Low']
        else:
            close_price = df['Close']
            high_price = df['High']
            low_price = df['Low']
        
        # Moving averages
        df['SMA_20'] = close_price.rolling(window=20).mean()
        df['EMA_20'] = close_price.ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = close_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        middle_band = close_price.rolling(window=20).mean()
        bb_std = close_price.rolling(window=20).std()
        df['BB_middle'] = middle_band
        df['BB_upper'] = middle_band + (2 * bb_std)
        df['BB_lower'] = middle_band - (2 * bb_std)
        
        # MACD
        exp1 = close_price.ewm(span=12, adjust=False).mean()
        exp2 = close_price.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Stochastic Oscillator
        low_min = low_price.rolling(window=14).min()
        high_max = high_price.rolling(window=14).max()
        df['STOCH_K'] = ((close_price - low_min) / (high_max - low_min)) * 100
        df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
        
        return df
    
    def fetch_fundamental_data(self, ticker):
        """Fetch fundamental data using Finnhub API"""
        try:
            financials = self.finnhub_client.company_basic_financials(
                ticker, 'all')
            metrics = {
                'P/E': financials.get('metric', {}).get('peBasicExclExtraTTM'),
                'EPS': financials.get('metric', {}).get('epsBasicExclExtraTTM'),
                'Debt_Ratio': financials.get('metric', {}).get('totalDebtToEquityQuarterly'),
                'Dividend_Yield': financials.get('metric', {}).get('dividendYieldIndicatedAnnual')
            }
            return metrics
        except Exception as e:
            print(f"Error fetching fundamental data for {ticker}: {e}")
            return {}

    def fetch_news_sentiment(self, ticker):
        """Fetch and analyze news sentiment"""
        try:
            # Format dates properly for Finnhub API
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            news = self.finnhub_client.company_news(
                ticker,
                _from=start_date,
                to=end_date
            )
            
            sentiments = []
            for article in news:
                blob = TextBlob(article['headline'])
                sentiments.append(blob.sentiment.polarity)
            
            return {
                'avg_sentiment': np.mean(sentiments) if sentiments else 0,
                'sentiment_std': np.std(sentiments) if sentiments else 0
            }
        except Exception as e:
            print(f"Error fetching news sentiment for {ticker}: {e}")
            return {'avg_sentiment': 0, 'sentiment_std': 0}

    def prepare_training_data(self, ticker, data):
        """Prepare data for ML model training"""
        try:
            print(f"\nPreparing data for {ticker}")
            print(f"Initial data columns: {data.columns}")
            
            # Calculate technical indicators first
            df = self.calculate_technical_indicators(data)
            
            # Now flatten the columns if they're multi-level
            if isinstance(df.columns, pd.MultiIndex):
                # Keep only the first level of the index for price data
                # and merge with the single-level technical indicators
                price_cols = df.loc[:, ['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                tech_cols = df.loc[:, [col for col in df.columns if col[0] not in ['Open', 'High', 'Low', 'Close', 'Volume']]].copy()
                
                # Flatten the technical indicator columns
                if isinstance(tech_cols.columns, pd.MultiIndex):
                    tech_cols.columns = tech_cols.columns.get_level_values(0)
                
                # Combine price and technical data
                df = pd.concat([price_cols, tech_cols], axis=1)
            
            # Reset index to handle any potential issues
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Add fundamental data
            fundamental_data = self.fetch_fundamental_data(ticker)
            for key, value in fundamental_data.items():
                df[key] = value
            
            # Add sentiment data
            sentiment_data = self.fetch_news_sentiment(ticker)
            for key, value in sentiment_data.items():
                df[key] = value
            
            # Fetch macro indicators
            macro_data = self.fetch_macro_indicators()
            
            # Resample macro data to daily frequency
            macro_data = macro_data.resample('D').ffill()
            
            # Align date ranges
            start_date = df.index.min()
            end_date = df.index.max()
            macro_data = macro_data.loc[start_date:end_date]
            
            print(f"\nFinal data structure:")
            print(f"DF columns: {df.columns.tolist()}")
            print(f"Macro columns: {macro_data.columns.tolist()}")
            
            # Merge the dataframes
            df = df.merge(macro_data, left_index=True, right_index=True, how='left')
            
            # Forward fill any missing values
            df = df.ffill().dropna()
            
            return df
            
        except Exception as e:
            print(f"Detailed error in prepare_training_data for {ticker}:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Data columns: {data.columns}")
            raise

    def fetch_macro_indicators(self):
        """Fetch macroeconomic indicators from FRED"""
        try:
            indicators = {
                'GDP': self.fred.get_series('GDP'),
                'INFLATION': self.fred.get_series('CPIAUCSL'),
                'UNEMPLOYMENT': self.fred.get_series('UNRATE'),
                'INTEREST_RATE': self.fred.get_series('FEDFUNDS')
            }
            
            # Create DataFrame with explicit date handling
            df = pd.DataFrame(indicators)
            df = df.reset_index()
            df.columns = ['Date'] + list(indicators.keys())
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            
            # Forward fill missing values
            df = df.ffill()
            
            print("\nMacro indicators structure:")
            print(f"Index type: {type(df.index)}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"Error in fetch_macro_indicators:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise

    def train_prediction_model(self, data, prediction_window=30):
        """Train ML model for price prediction"""
        features = data.drop(['Close'], axis=1)
        target = data['Close'].shift(-prediction_window)
        
        features = features.dropna()
        target = target.dropna()
        features = features[:len(target)]
        
        train_size = int(len(features) * 0.8)
        X_train = features[:train_size]
        y_train = target[:train_size]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        return model

    def generate_enhanced_recommendations(self, tickers):
        """Generate enhanced trading recommendations"""
        recommendations = {}
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, period="1y", interval="1d", progress=False)
                prepared_data = self.prepare_training_data(ticker, data)
                model = self.train_prediction_model(prepared_data)
                
                latest_data = prepared_data.iloc[-1:]
                prediction = model.predict(latest_data.drop(['Close'], axis=1))
                
                current_price = data['Close'].iloc[-1]
                expected_return = (prediction[0] - current_price) / current_price
                
                fundamental_score = sum(self.fetch_fundamental_data(ticker).values())
                sentiment_score = self.fetch_news_sentiment(ticker)['avg_sentiment']
                
                total_score = (expected_return * 0.5 + 
                             fundamental_score * 0.3 + 
                             sentiment_score * 0.2)
                
                recommendations[ticker] = total_score
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return recommendations

    def optimize_portfolio(self, recommendations, risk_tolerance=0.5):
        """Optimize portfolio based on enhanced recommendations"""
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        portfolio = {}
        total_score = sum(abs(score) for _, score in sorted_recs)
        
        for ticker, score in sorted_recs:
            if score > 0:
                weight = (score / total_score) * risk_tolerance
                portfolio[ticker] = weight
            
        total_weight = sum(portfolio.values())
        if total_weight > 0:
            portfolio = {k: v/total_weight for k, v in portfolio.items()}
            
        return portfolio

def fetch_all_tickers():
    """Fetch all tickers for analysis"""
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    top_tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC', 'CSCO']
    top_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO', 'VTV', 'VUG', 'VOO']
    top_comodities_etfs = ['GLD', 'SLV', 'USO', 'UNG', 'PPLT', 'PALL', 'WEAT', 'CORN', 'DBA', 'DBB', 'DBC', 'DBO', 'DBP']
    
    #return islamic_us_etfs + top_tech + top_etfs + top_comodities_etfs
    return islamic_us_etfs

def main():
    # Initialize API keys (replace with your actual API keys)
    api_keys = {
        'finnhub': 'cu2ha6pr01qh0l7ha4mgcu2ha6pr01qh0l7ha4n0',
        'fred': 'cd852e18eff164cf69663b2b638f9d1e'
    }
    
    # Initialize analyzer
    analyzer = EnhancedStockAnalysis(api_keys)
    
    # Fetch tickers
    tickers = fetch_all_tickers()
    
    print(f"Analyzing {len(tickers)} tickers...")
    
    # Generate recommendations
    recommendations = analyzer.generate_enhanced_recommendations(tickers)
    
    # Optimize portfolio
    optimized_portfolio = analyzer.optimize_portfolio(recommendations)
    
    # Display results
    print("\nOptimized Portfolio Allocation:")
    for ticker, weight in sorted(optimized_portfolio.items(), key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%}")

if __name__ == "__main__":
    main()