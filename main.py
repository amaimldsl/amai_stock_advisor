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
        """Initialize with API keys and metric mappings"""
        self.finnhub_client = finnhub.Client(api_key=api_keys['finnhub'])
        self.fred = Fred(api_key=api_keys['fred'])
        self.scaler = MinMaxScaler()
        
        # Define metric mappings as class attribute
        self.metric_mappings = {
            'P/E': 'peBasicExclExtraItems',
            'EPS': 'epsTTM',
            'Debt_Ratio': 'totalDebtToEquity',
            'Dividend_Yield': 'dividendYield',
            'ROE': 'returnOnEquityTTM',
            'Current_Ratio': 'currentRatio',
            'Gross_Margin': 'grossMarginTTM',
            'Operating_Margin': 'operatingMarginTTM'
        }
        
        # Alternative metrics for fallback
        self.alternative_metrics = {
            'EPS': ['eps', 'epsGrowthTTM', 'epsInclExtraItemsTTM'],
            'P/E': ['peNormalizedAnnual', 'peExclExtraTTM'],
            'Debt_Ratio': ['debtToAssets', 'longTermDebtToEquity'],
            'ROE': ['roeTTM', 'roeRfy'],
            'Current_Ratio': ['quickRatio']
        }

    def _safe_get_metric(self, financials, metric_name):
        """Safely extract a metric from financials data with enhanced error handling"""
        try:
            if 'metric' not in financials:
                print(f"No 'metric' field in financials data")
                return None
                
            value = financials['metric'].get(metric_name)
            
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError) as e:
                    print(f"Error converting {metric_name} value '{value}' to float: {e}")
                    return None
            else:
                print(f"Metric {metric_name} not found in response")
                return None
                
        except Exception as e:
            print(f"Error extracting metric {metric_name}: {e}")
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
        """Enhanced technical indicators with error recovery"""
        try:
            prices = self.get_price_columns(data, ticker)
            df_new = pd.DataFrame(index=data.index)
            
            # Basic price data
            for col_name, series in prices.items():
                df_new[col_name] = series
                
            # Technical indicators with individual try-except blocks
            try:
                df_new['SMA_20'] = prices['Close'].rolling(window=20).mean()
            except Exception as e:
                print(f"Error calculating SMA: {e}")
                df_new['SMA_20'] = None
                
            try:
                df_new['EMA_20'] = prices['Close'].ewm(span=20, adjust=False).mean()
            except Exception as e:
                print(f"Error calculating EMA: {e}")
                df_new['EMA_20'] = None
                
            try:
                # RSI with better error handling
                delta = prices['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_new['RSI'] = 100 - (100 / (1 + rs))
            except Exception as e:
                print(f"Error calculating RSI: {e}")
                df_new['RSI'] = None
                
            return df_new
            
        except Exception as e:
            print(f"Critical error in technical indicators: {e}")
            return pd.DataFrame()
                    
    def fetch_fundamental_data(self, ticker):
        """Fetch fundamental data with enhanced fallback options and yfinance backup"""
        try:
            print(f"\nFetching fundamental data for {ticker}...")
            
            # First check if ticker is an ETF
            if self.is_etf(ticker):
                return self.fetch_etf_data(ticker)
            
            # Initialize metrics dictionary
            metrics = {}
            
            # Try Finnhub first
            financials = self.finnhub_client.company_basic_financials(ticker, 'all')
            
            # If Finnhub fails or misses metrics, use yfinance as backup
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            # Enhanced metric mapping with fallbacks
            metric_sources = {
                'P/E': [
                    ('finnhub', 'peBasicExclExtraItems'),
                    ('finnhub', 'peNormalizedAnnual'),
                    ('yfinance', 'forwardPE'),
                    ('yfinance', 'trailingPE')
                ],
                'EPS': [
                    ('finnhub', 'epsTTM'),
                    ('yfinance', 'trailingEps'),
                    ('yfinance', 'forwardEps')
                ],
                'Debt_Ratio': [
                    ('finnhub', 'totalDebtToEquity'),
                    ('finnhub', 'debtToAssets'),
                    ('yfinance', 'debtToEquity')
                ],
                'ROE': [
                    ('finnhub', 'returnOnEquityTTM'),
                    ('finnhub', 'roeTTM'),
                    ('yfinance', 'returnOnEquity')
                ],
                'Current_Ratio': [
                    ('finnhub', 'currentRatio'),
                    ('finnhub', 'quickRatio'),
                    ('yfinance', 'currentRatio')
                ],
                'Gross_Margin': [
                    ('finnhub', 'grossMarginTTM'),
                    ('yfinance', 'grossMargins')
                ],
                'Operating_Margin': [
                    ('finnhub', 'operatingMarginTTM'),
                    ('yfinance', 'operatingMargins')
                ],
                'Revenue_Growth': [
                    ('finnhub', 'revenueGrowthTTM'),
                    ('yfinance', 'revenueGrowth')
                ],
                'Profit_Margin': [
                    ('finnhub', 'netProfitMarginTTM'),
                    ('yfinance', 'profitMargins')
                ]
            }
            
            # Try to get each metric from multiple sources
            for metric_name, sources in metric_sources.items():
                value = None
                for source, key in sources:
                    try:
                        if source == 'finnhub':
                            if financials and 'metric' in financials:
                                value = financials['metric'].get(key)
                        elif source == 'yfinance':
                            value = info.get(key)
                            
                        if value is not None:
                            try:
                                value = float(value)
                                print(f"{ticker} {metric_name}: {value} (from {source})")
                                metrics[metric_name] = value
                                break
                            except (ValueError, TypeError):
                                continue
                    except Exception as e:
                        print(f"Error getting {metric_name} from {source}: {e}")
                        continue
                
                if value is None:
                    print(f"Metric {metric_name} not found for {ticker}")
            
            # Add market cap and volume metrics
            try:
                if info.get('marketCap'):
                    metrics['Market_Cap'] = float(info['marketCap'])
                if info.get('volume'):
                    metrics['Volume'] = float(info['volume'])
            except Exception as e:
                print(f"Error getting market data: {e}")
            
            # Calculate additional derived metrics
            if 'Market_Cap' in metrics and 'Volume' in metrics:
                try:
                    metrics['Volume_to_Market_Cap'] = metrics['Volume'] / metrics['Market_Cap']
                except Exception as e:
                    print(f"Error calculating derived metrics: {e}")
            
            return metrics
            
        except Exception as e:
            print(f"Error in fetch_fundamental_data for {ticker}: {e}")
            return {}        
                    
    def is_etf(self, ticker):
        """Helper function to identify ETFs"""
        try:
            # Get basic info from yfinance
            info = yf.Ticker(ticker).info
            return info.get('quoteType', '').upper() == 'ETF'
        except:
            # If error, check against known ETF lists
            known_etfs = {'SPY', 'QQQ', 'SPUS', 'HLAL', 'VTI', 'VOO'}  # Add more as needed
            return ticker in known_etfs

    def fetch_etf_data(self, ticker):
        """Specialized ETF data fetching"""
        try:
            etf = yf.Ticker(ticker)
            info = etf.info
            
            metrics = {
                'Expense_Ratio': info.get('expenseRatio', None),
                'AUM': info.get('totalAssets', None),
                'NAV': info.get('previousClose', None),  # Use previous close as NAV approximation
                'Dividend_Yield': info.get('dividendYield', None),
                'Beta': info.get('beta3Year', None),
                'Volume': info.get('averageVolume', None)
            }
            
            # Filter out None values and convert to float where possible
            cleaned_metrics = {}
            for k, v in metrics.items():
                if v is not None:
                    try:
                        cleaned_metrics[k] = float(v)
                    except (TypeError, ValueError):
                        cleaned_metrics[k] = v
            
            return cleaned_metrics
            
        except Exception as e:
            print(f"Error fetching ETF data for {ticker}: {e}")
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
    
    def normalize_fundamental_score(self, score, metric_count, is_etf):
        """Normalize fundamental scores to a reasonable range"""
        try:
            if score == 0 or metric_count == 0:
                return 0
                
            # For ETFs, apply log normalization to handle large AUM values
            if is_etf:
                normalized = np.log(abs(score) + 1) / metric_count
                # Cap the maximum value at 1
                return min(normalized, 1) * (1 if score > 0 else -1)
            
            # For stocks, normalize based on typical fundamental ratio ranges
            else:
                # Assume typical fundamental ratios are in 0-100 range
                normalized = score / (100 * metric_count)
                # Cap at [-1, 1] range
                return max(min(normalized, 1), -1)
                
        except Exception as e:
            print(f"Error in normalize_fundamental_score: {e}")
            return 0
    
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
        """Generate recommendations with properly normalized scores"""
        recommendations = {}
        
        for ticker in tickers:
            try:
                # Previous data preparation code...
                data = yf.download(ticker, period="1y", interval="1d", progress=False)
                if data.empty:
                    continue
                    
                prepared_data = self.prepare_training_data(ticker, data)
                if prepared_data.empty:
                    continue
                    
                # Calculate expected return
                model = self.train_prediction_model(prepared_data)
                latest_data = prepared_data.iloc[-1:]
                prediction = model.predict(latest_data.drop(['Close'], axis=1))
                current_price = prepared_data['Close'].iloc[-1]
                expected_return = (prediction[0] - current_price) / current_price
                
                # Get fundamental data and normalize scores
                fundamental_data = self.fetch_fundamental_data(ticker)
                is_etf = self.is_etf(ticker)
                
                # Calculate fundamental score
                fundamental_score = sum(value for value in fundamental_data.values() 
                                    if value is not None and isinstance(value, (int, float)))
                valid_fundamentals = len([v for v in fundamental_data.values() 
                                        if v is not None and isinstance(v, (int, float))])
                
                # Normalize fundamental score
                normalized_fundamental_score = self.normalize_fundamental_score(
                    fundamental_score, valid_fundamentals, is_etf)
                
                # Get sentiment score (already normalized between -1 and 1)
                sentiment_data = self.fetch_news_sentiment(ticker)
                sentiment_score = sentiment_data.get('avg_sentiment', 0)
                
                # Calculate total score with normalized components
                total_score = (
                    expected_return * 0.5 + 
                    normalized_fundamental_score * 0.3 + 
                    sentiment_score * 0.2
                )
                
                recommendations[ticker] = total_score
                
                print(f"\nDetailed scoring for {ticker}:")
                print(f"Expected Return: {expected_return:.4f}")
                print(f"Fundamental Score: {normalized_fundamental_score:.4f} (from {valid_fundamentals} metrics)")
                print(f"Sentiment Score: {sentiment_score:.4f}")
                print(f"Total Score: {total_score:.4f}")
                
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                continue
        
        return recommendations
    def optimize_portfolio(self, recommendations, risk_tolerance=0.5, include_commodities=True):
        """
        Optimize portfolio with buy/sell recommendations and commodity hedging
        """
        try:
            # Sort recommendations by absolute score value to properly identify both positive and negative performers
            sorted_recs = sorted(recommendations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Separate positive and negative recommendations
            buy_recs = [(t, s) for t, s in sorted_recs if s > 0]
            sell_recs = [(t, s) for t, s in sorted_recs if s < 0]
            
            # Get top 10 buy recommendations
            top_buys = {}
            total_buy_score = sum(score for _, score in buy_recs[:10])
            if total_buy_score > 0:
                for ticker, score in buy_recs[:10]:
                    weight = (score / total_buy_score) * risk_tolerance
                    top_buys[ticker] = weight
            
            # Get top 10 sell recommendations - ensure we look at negative scores
            top_sells = {}
            total_sell_score = sum(abs(score) for _, score in sell_recs[:10])
            if total_sell_score > 0:
                for ticker, score in sell_recs[:10]:
                    weight = (abs(score) / total_sell_score) * (1 - risk_tolerance)
                    top_sells[ticker] = weight
            
            # Calculate portfolio metrics
            buy_tickers = list(top_buys.keys())
            portfolio_volatility = self.calculate_portfolio_volatility(buy_tickers)
            market_correlation = self.calculate_market_correlation(buy_tickers)
            
            print(f"\nPortfolio Metrics:")
            print(f"Volatility: {portfolio_volatility:.4f}")
            print(f"Market Correlation: {market_correlation:.4f}")
            
            # Enhanced commodity analysis and allocation
            commodity_allocation = {}
            if include_commodities:
                commodity_allocation = self.analyze_commodity_hedging(
                    portfolio_volatility, 
                    market_correlation
                )
            
            return {
                'buy_recommendations': top_buys,
                'sell_recommendations': top_sells,
                'commodity_hedging': commodity_allocation
            }
            
        except Exception as e:
            print(f"Error in optimize_portfolio: {e}")
            return {
                'buy_recommendations': {},
                'sell_recommendations': {},
                'commodity_hedging': {}
            }


    def calculate_portfolio_volatility(self, tickers):
        """Calculate portfolio volatility with proper handling of pandas Series"""
        try:
            if not tickers:  # Handle empty tickers list
                return 0.0
                
            # Download historical data
            data = yf.download(tickers, period="1y", interval="1d")['Close']
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # If we have multiple tickers, calculate portfolio returns
            if isinstance(returns, pd.DataFrame):
                # Equal-weighted portfolio for simplicity
                weights = np.array([1/len(tickers)] * len(tickers))
                portfolio_returns = returns.dot(weights)
                volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            else:
                # Single ticker case
                volatility = returns.std() * np.sqrt(252)
                
            return float(volatility)  # Ensure we return a float
            
        except Exception as e:
            print(f"Error calculating portfolio volatility: {e}")
            return 0.0





    def calculate_market_correlation(self, tickers):
        """Calculate correlation with market (SPY) with proper error handling"""
        try:
            if not tickers:  # Handle empty tickers list
                return 0.0
                
            # Add SPY to the list of tickers if not already present
            if 'SPY' not in tickers:
                all_tickers = tickers + ['SPY']
            else:
                all_tickers = tickers
                
            # Download historical data
            data = yf.download(all_tickers, period="1y", interval="1d")['Close']
            returns = data.pct_change().dropna()
            
            # Calculate correlation with SPY
            if isinstance(returns, pd.DataFrame):
                correlations = returns.corr()['SPY'].drop('SPY')  # Remove SPY's correlation with itself
                return float(correlations.mean())  # Average correlation
            else:
                return 0.0  # Return 0 if we only have SPY
                
        except Exception as e:
            print(f"Error calculating market correlation: {e}")
            return 0.0
    def analyze_commodity_hedging(self, portfolio_volatility, market_correlation):
        """Enhanced commodity ETF hedging analysis with proper column handling"""
        try:
            commodity_etfs = {
                'GLD': 'gold',
                'SLV': 'silver',
                'USO': 'oil',
                'DBC': 'broad_commodities',
                'CORN': 'corn',
                'WEAT': 'wheat',
                'UNG': 'natural_gas'
            }
            
            hedging_allocation = {}
            
            # Download commodity ETF data - use Close price instead of Adj Close
            commodity_data = yf.download(list(commodity_etfs.keys()), period="6mo")['Close']
            returns = commodity_data.pct_change().dropna()
            volatilities = returns.std() * np.sqrt(252)
            
            # Calculate momentum scores
            momentum = (commodity_data.iloc[-1] / commodity_data.iloc[-126] - 1)
            
            for etf in commodity_etfs:
                if etf in volatilities.index and etf in momentum.index:
                    vol_score = 1 - (volatilities[etf] / volatilities.max())
                    mom_score = (momentum[etf] - momentum.min()) / (momentum.max() - momentum.min())
                    score = (vol_score + mom_score) / 2
                    
                    if score > 0.5:
                        if market_correlation > 0.7:
                            hedging_allocation[etf] = score * 0.05
                        else:
                            hedging_allocation[etf] = score * 0.03
            
            # Normalize allocations
            if hedging_allocation:
                total_allocation = sum(hedging_allocation.values())
                max_commodity_exposure = 0.20
                if total_allocation > 0:
                    hedging_allocation = {k: (v/total_allocation) * max_commodity_exposure 
                                    for k, v in hedging_allocation.items()}
            
            return hedging_allocation
            
        except Exception as e:
            print(f"Error in analyze_commodity_hedging: {e}")
            return {}



def fetch_all_tickers():
    """Fetch all tickers for analysis"""
    islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    top_tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC', 'CSCO']
    top_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO', 'VTV', 'VUG', 'VOO']
    commodity_etfs = ['GLD', 'SLV', 'USO', 'UNG', 'PPLT', 'PALL', 'WEAT', 'CORN', 'DBA', 'DBB', 'DBC', 'DBO', 'DBP']
    
    # Return all categories of tickers
    return islamic_us_etfs + top_tech + top_etfs + commodity_etfs
        
def main():
    # Initialize API keys (replace with your actual API keys)
    api_keys = {
        'finnhub': 'cu2ha6pr01qh0l7ha4mgcu2ha6pr01qh0l7ha4n0',
        'fred': 'cd852e18eff164cf69663b2b638f9d1e'
    }
    
       # Initialize analyzer
    analyzer = EnhancedStockAnalysis(api_keys)
    
    # Fetch all tickers including commodity ETFs
    tickers = fetch_all_tickers()
    
    print(f"Analyzing {len(tickers)} tickers...")
    
    # Generate recommendations
    recommendations = analyzer.generate_enhanced_recommendations(tickers)
    
    # Optimize portfolio with new features
    portfolio = analyzer.optimize_portfolio(recommendations, risk_tolerance=0.6)
    
    # Display results
    print("\nTop 10 Buy Recommendations:")
    for ticker, weight in sorted(portfolio['buy_recommendations'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%} allocation (Expected gain potential)")
    
    print("\nTop 10 Sell Recommendations:")
    for ticker, weight in sorted(portfolio['sell_recommendations'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%} allocation (Risk exposure)")
    
    if portfolio['commodity_hedging']:
        print("\nRecommended Commodity Hedging:")
        for ticker, weight in portfolio['commodity_hedging'].items():
            print(f"{ticker}: {weight:.2%} allocation (Risk hedge)")

if __name__ == "__main__":
    main()    # Initialize API keys (replace with your actual API keys)
    api_keys = {
        'finnhub': 'cu2ha6pr01qh0l7ha4mgcu2ha6pr01qh0l7ha4n0',
        'fred': 'cd852e18eff164cf69663b2b638f9d1e'
    }
    
    # Initialize analyzer
    analyzer = EnhancedStockAnalysis(api_keys)
    
    # Fetch all tickers including commodity ETFs
    tickers = fetch_all_tickers()
    
    print(f"Analyzing {len(tickers)} tickers...")
    
    # Generate recommendations
    recommendations = analyzer.generate_enhanced_recommendations(tickers)
    
    # Optimize portfolio with enhanced features
    portfolio = analyzer.optimize_portfolio(recommendations, risk_tolerance=0.6, include_commodities=True)
    
    # Display results
    print("\nTop Buy Recommendations:")
    for ticker, weight in sorted(portfolio['buy_recommendations'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%} allocation")
    
    print("\nTop Sell Recommendations:")
    for ticker, weight in sorted(portfolio['sell_recommendations'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%} allocation")
    
    print("\nCommodity Hedging Allocation:")
    for ticker, weight in sorted(portfolio['commodity_hedging'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{ticker}: {weight:.2%} allocation")

if __name__ == "__main__":
    main()