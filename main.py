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

    def _safe_get_metric(self, financials, metric_name):
        """Safely extract a metric from financials data"""
        try:
            value = financials.get('metric', {}).get(metric_name)
            if value is not None:
                return float(value)
            return None
        except (TypeError, ValueError) as e:
            print(f"Error converting metric {metric_name}: {e}")
            return None

    def get_price_columns(self, data, ticker):
        """Helper function to extract price columns regardless of DataFrame structure"""
        if isinstance(data.columns, pd.MultiIndex):
            return {
                'Close': data[('Close', ticker)],
                'High': data[('High', ticker)],
                'Low': data[('Low', ticker)],
                'Open': data[('Open', ticker)],
                'Volume': data[('Volume', ticker)]
            }
        else:
            return {
                'Close': data['Close'],
                'High': data['High'],
                'Low': data['Low'],
                'Open': data['Open'],
                'Volume': data['Volume']
            }
        
    def calculate_technical_indicators(self, data, ticker):
        """Calculate technical indicators using pandas with explicit ticker handling"""
        try:
            # Get price columns regardless of DataFrame structure
            prices = self.get_price_columns(data, ticker)
            
            # Create a new dataframe with single-level columns
            df_new = pd.DataFrame(index=data.index)
            
            # Add base price columns
            for col_name, series in prices.items():
                df_new[col_name] = series
            
            # Calculate technical indicators using the correct price series
            close_prices = prices['Close']
            high_prices = prices['High']
            low_prices = prices['Low']
            
            # Moving averages
            df_new['SMA_20'] = close_prices.rolling(window=20).mean()
            df_new['EMA_20'] = close_prices.ewm(span=20, adjust=False).mean()
            
            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_new['RSI'] = 100 - (100 / (1 + rs))
            
            # Rest of indicators remain the same...
            # [Previous technical indicator calculations]
            
            return df_new
            
        except Exception as e:
            print(f"Error in calculate_technical_indicators for {ticker}:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise
    
    def fetch_fundamental_data(self, ticker):
        """Fetch fundamental data with enhanced error handling"""
        try:
            financials = self.finnhub_client.company_basic_financials(ticker, 'all')
            if not financials or not isinstance(financials, dict):
                print(f"No financial data returned for {ticker}")
                return {}
                
            metrics = {}
            metric_mappings = {
                'P/E': 'peBasicExclExtraTTM',
                'EPS': 'epsBasicExclExtraTTM',
                'Debt_Ratio': 'totalDebtToEquityQuarterly',
                'Dividend_Yield': 'dividendYieldIndicatedAnnual'
            }
            
            for display_name, api_name in metric_mappings.items():
                value = self._safe_get_metric(financials, api_name)
                metrics[display_name] = value
                print(f"{ticker} {display_name}: {value}")
            
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
        """Prepare data for ML model training with enhanced error handling"""
        try:
            print(f"\nPreparing data for {ticker}")
            print(f"Initial data shape: {data.shape}")
            
            # Step 1: Calculate technical indicators
            df = self.calculate_technical_indicators(data, ticker)
            print(f"After technical indicators shape: {df.shape}")
            
            # Step 2: Ensure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Step 3: Add fundamental data
            fundamental_data = self.fetch_fundamental_data(ticker)
            if fundamental_data:
                for key, value in fundamental_data.items():
                    if value is not None:  # Only add non-None values
                        df[key] = value
            print(f"After fundamental data shape: {df.shape}")
            
            # Step 4: Add sentiment data
            sentiment_data = self.fetch_news_sentiment(ticker)
            if sentiment_data:
                for key, value in sentiment_data.items():
                    df[key] = value
            print(f"After sentiment data shape: {df.shape}")
            
            # Step 5: Fetch macro indicators
            try:
                macro_data = self.fetch_macro_indicators()
                
                # Resample macro data to daily frequency if needed
                if not macro_data.empty:
                    macro_data = macro_data.resample('D').ffill()
                    
                    # Align date ranges
                    start_date = df.index.min()
                    end_date = df.index.max()
                    macro_data = macro_data.loc[start_date:end_date]
                    
                    # Merge the dataframes
                    df = df.merge(macro_data, left_index=True, right_index=True, how='left')
                    print(f"After macro data merge shape: {df.shape}")
            except Exception as macro_error:
                print(f"Warning: Error fetching macro data: {macro_error}")
                # Continue without macro data
            
            # Step 6: Handle missing values
            # First, print info about missing values
            print("\nMissing values before cleaning:")
            print(df.isnull().sum())
            
            # Forward fill missing values
            df = df.ffill()
            
            # Fill any remaining NaN with 0
            df = df.fillna(0)
            
            # Drop any columns that are all zeros
            df = df.loc[:, (df != 0).any(axis=0)]
            
            # Ensure we have the required 'Close' column
            if 'Close' not in df.columns:
                # Try to find it in case it's named differently
                close_columns = [col for col in df.columns if 'close' in col.lower()]
                if close_columns:
                    df['Close'] = df[close_columns[0]]
                else:
                    raise ValueError(f"No 'Close' price column found for {ticker}")
            
            print(f"\nFinal data shape: {df.shape}")
            print(f"Final columns: {df.columns.tolist()}")
            
            if df.empty:
                raise ValueError(f"Prepared data is empty for {ticker}")
                
            return df
            
        except Exception as e:
            print(f"\nDetailed error in prepare_training_data for {ticker}:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            if 'df' in locals():
                print(f"DataFrame shape at error: {df.shape}")
                print(f"DataFrame columns at error: {df.columns.tolist()}")
                print("\nSample of data at error:")
                print(df.head())
                print("\nMissing values at error:")
                print(df.isnull().sum())
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def validate_data(self, ticker, data):
        """Validate data before processing"""
        try:
            # Check if data is empty
            if data.empty:
                print(f"Empty data received for {ticker}")
                return False
            
            # Check for minimum required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if isinstance(data.columns, pd.MultiIndex):
                # For multi-index columns from yfinance
                available_columns = [col[0] for col in data.columns]
            else:
                available_columns = data.columns.tolist()
                
            missing_columns = [col for col in required_columns 
                            if not any(col.lower() == c.lower() for c in available_columns)]
            
            if missing_columns:
                print(f"Missing required columns for {ticker}: {missing_columns}")
                return False
            
            # Check for minimum data points (e.g., need at least 30 days for meaningful analysis)
            if len(data) < 30:
                print(f"Insufficient data points for {ticker}: {len(data)} < 30")
                return False
            
            # Check for too many missing values
            missing_pct = data.isnull().sum() / len(data)
            if any(missing_pct > 0.5):
                print(f"Too many missing values for {ticker}")
                print(missing_pct[missing_pct > 0.5])
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating data for {ticker}: {e}")
            return False
    
    
    
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
        try:
            # Ensure 'Close' column exists
            if 'Close' not in data.columns:
                raise KeyError(f"'Close' column not found. Available columns: {data.columns}")
            
            # Create features and target
            features = data.drop(['Close'], axis=1, errors='ignore')  # Use errors='ignore' to handle if column doesn't exist
            target = data['Close'].shift(-prediction_window)
            
            # Clean data
            features = features.dropna()
            target = target.dropna()
            
            # Align lengths
            min_len = min(len(features), len(target))
            features = features[:min_len]
            target = target[:min_len]
            
            # Split data
            train_size = int(len(features) * 0.8)
            X_train = features[:train_size]
            y_train = target[:train_size]
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            return model
            
        except Exception as e:
            print(f"Error in train_prediction_model:")
            print(f"Features shape: {features.shape if 'features' in locals() else 'Not created'}")
            print(f"Target shape: {target.shape if 'target' in locals() else 'Not created'}")
            print(f"Available columns: {data.columns}")
            raise


    def generate_enhanced_recommendations(self, tickers):
        """Generate enhanced trading recommendations with proper None handling"""
        recommendations = {}
        
        for ticker in tickers:
            try:
                # Download data
                data = yf.download(ticker, period="1y", interval="1d", progress=False)
                if data.empty:
                    print(f"No data downloaded for {ticker}")
                    continue
                
                # Prepare data
                prepared_data = self.prepare_training_data(ticker, data)
                if prepared_data.empty:
                    print(f"No prepared data for {ticker}")
                    continue
                
                # Train model and generate prediction
                model = self.train_prediction_model(prepared_data)
                latest_data = prepared_data.iloc[-1:]
                prediction = model.predict(latest_data.drop(['Close'], axis=1))
                
                # Calculate expected return
                current_price = prepared_data['Close'].iloc[-1]
                expected_return = (prediction[0] - current_price) / current_price
                
                # Safely calculate fundamental score
                fundamental_data = self.fetch_fundamental_data(ticker)
                fundamental_score = 0
                valid_fundamentals = 0
                
                for value in fundamental_data.values():
                    if value is not None:
                        try:
                            fundamental_score += float(value)
                            valid_fundamentals += 1
                        except (TypeError, ValueError) as e:
                            print(f"Error converting fundamental value: {e}")
                            continue
                
                # Normalize fundamental score if we have valid data
                if valid_fundamentals > 0:
                    fundamental_score = fundamental_score / valid_fundamentals
                
                # Safely get sentiment score
                sentiment_data = self.fetch_news_sentiment(ticker)
                sentiment_score = sentiment_data.get('avg_sentiment', 0)
                
                # Calculate total score with proper weights
                if valid_fundamentals > 0:
                    total_score = (
                        expected_return * 0.5 + 
                        fundamental_score * 0.3 + 
                        sentiment_score * 0.2
                    )
                else:
                    total_score = (
                        expected_return * 0.7 + 
                        sentiment_score * 0.3
                    )
                
                recommendations[ticker] = total_score
                
                # Print detailed scoring information
                print(f"\nDetailed scoring for {ticker}:")
                print(f"Expected Return: {expected_return:.4f}")
                print(f"Fundamental Score: {fundamental_score:.4f} (from {valid_fundamentals} metrics)")
                print(f"Sentiment Score: {sentiment_score:.4f}")
                print(f"Total Score: {total_score:.4f}")
                
            except Exception as e:
                print(f"\nDetailed error processing {ticker}:")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
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
    return top_tech




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