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
import logging
import contextlib

import yfinance as yf


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockAnalysis')

class EnhancedStockAnalysis:
    
    def __init__(self, api_keys):
        """Initialize with API keys and configuration"""
        if not isinstance(api_keys, dict) or not all(k in api_keys for k in ['finnhub', 'fred']):
            raise ValueError("api_keys must be a dict containing 'finnhub' and 'fred' keys")
            
        self.finnhub_client = finnhub.Client(api_key=api_keys['finnhub'])
        self.fred = Fred(api_key=api_keys['fred'])
        self.scaler = MinMaxScaler()
        self.logger = logging.getLogger('StockAnalysis')
        
        # Configuration
        self.config = {
            'data_periods': {
                'short_term': '1mo',
                'medium_term': '6mo',
                'long_term': '1y'
            },
            'score_weights': {
                'technical': 0.4,
                'fundamental': 0.4,
                'sentiment': 0.2
            }
        }


    def _get_cached_data(self, key):
        """Get cached data if valid"""
        if key in self._cache and self._cache_ttl[key] > datetime.now():
            return self._cache[key]
        return None
        
    def _set_cached_data(self, key, data):
        """Cache data with TTL"""
        self._cache[key] = data
        self._cache_ttl[key] = datetime.now() + self.CACHE_DURATION

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


    def analyze_market_conditions(self):
        """Analyze market conditions with improved volatility calculation"""
        try:
            spy_data = yf.download('SPY', period=self.config['data_periods']['medium_term'])
            vix_data = yf.download('^VIX', period=self.config['data_periods']['short_term'])
            
            # Calculate returns and volatility - use .iloc[0] for proper Series element access
            returns = spy_data['Close'].pct_change()
            volatility = float(np.sqrt(252) * returns.rolling(window=21).std().iloc[-1].iloc[0])
            
            # Calculate trend strength - use .iloc[0] for proper Series element access
            sma_20 = spy_data['Close'].rolling(window=20).mean()
            sma_50 = spy_data['Close'].rolling(window=50).mean()
            trend_strength = float((sma_20.iloc[-1].iloc[0] / sma_50.iloc[-1].iloc[0]) - 1)
            
            # Get VIX level with proper Series element access
            vix_level = float(vix_data['Close'].iloc[-1].iloc[0])
            
            # Calculate market stress
            market_stress = self._calculate_market_stress(
                volatility=volatility,
                vix=vix_level,
                trend_strength=trend_strength
            )
            
            return {
                'market_stress': market_stress,
                'current_volatility': volatility,
                'vix_level': vix_level,
                'market_trend': trend_strength * 100  # Convert to percentage
            }
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return {
                'market_stress': 0.5,  # Default to moderate stress
                'current_volatility': 0.2,
                'vix_level': 20,
                'market_trend': 0
            }








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
        """Fetch fundamental data with improved metric extraction"""
        try:
            metrics = {}
            
            # Get data from Finnhub
            financials = self.finnhub_client.company_basic_financials(ticker, 'all')
            
            # Get data from yfinance as backup
            yf_ticker = yf.Ticker(ticker)
            try:
                info = yf_ticker.info
            except:
                info = {}

            # Metric mapping dictionary
            metric_mappings = {
                'P/E': {
                    'finnhub': 'peBasicExcl',
                    'yfinance': 'forwardPE',
                    'fallback': 'trailingPE'
                },
                'EPS': {
                    'finnhub': 'epsBasicExcl',
                    'yfinance': 'forwardEps',
                    'fallback': 'trailingEps'
                },
                'Debt_Ratio': {
                    'finnhub': 'totalDebt/totalAssets',
                    'yfinance': 'debtToEquity'
                },
                'ROE': {
                    'finnhub': 'roeRfy',
                    'yfinance': 'returnOnEquity'
                },
                'Current_Ratio': {
                    'finnhub': 'currentRatio',
                    'yfinance': 'currentRatio'
                },
                'Gross_Margin': {
                    'finnhub': 'grossMargin',
                    'yfinance': 'grossMargins'
                },
                'Operating_Margin': {
                    'finnhub': 'operatingMargin',
                    'yfinance': 'operatingMargins'
                },
                'Revenue_Growth': {
                    'finnhub': 'revenueGrowth',
                    'yfinance': 'revenueGrowth'
                },
                'Profit_Margin': {
                    'finnhub': 'netProfitMargin',
                    'yfinance': 'profitMargins'
                }
            }

            # Extract metrics with detailed logging
            print(f"\nFetching fundamental data for {ticker}:")
            
            for metric_name, sources in metric_mappings.items():
                try:
                    # Try Finnhub first
                    if 'finnhub' in sources and 'metric' in financials:
                        value = financials['metric'].get(sources['finnhub'])
                        if value is not None:
                            metrics[metric_name] = float(value)
                            print(f"{metric_name} (Finnhub): {value}")
                            continue

                    # Try primary yfinance source
                    if 'yfinance' in sources and sources['yfinance'] in info:
                        value = info[sources['yfinance']]
                        if value is not None:
                            metrics[metric_name] = float(value)
                            print(f"{metric_name} (YFinance): {value}")
                            continue

                    # Try fallback yfinance source if available
                    if 'fallback' in sources and sources['fallback'] in info:
                        value = info[sources['fallback']]
                        if value is not None:
                            metrics[metric_name] = float(value)
                            print(f"{metric_name} (YFinance fallback): {value}")
                            continue

                    print(f"{metric_name}: Not available")

                except (TypeError, ValueError) as e:
                    print(f"Error processing {metric_name}: {e}")
                    continue

            # Calculate additional ratios if possible
            if 'totalDebt' in info and 'totalAssets' in info and 'Debt_Ratio' not in metrics:
                try:
                    debt_ratio = float(info['totalDebt']) / float(info['totalAssets'])
                    metrics['Debt_Ratio'] = debt_ratio
                    print(f"Debt_Ratio (calculated): {debt_ratio}")
                except:
                    pass

            # Log summary of available metrics
            print(f"\nSummary for {ticker}:")
            print(f"Found {len(metrics)} metrics out of {len(metric_mappings)} possible metrics")
            print("Available metrics:", list(metrics.keys()))
            print("Missing metrics:", set(metric_mappings.keys()) - set(metrics.keys()))

            return metrics

        except Exception as e:
            print(f"Error fetching fundamental data for {ticker}: {e}")
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
        """Enhanced data preparation with proper MultiIndex handling"""
        try:
            print(f"\nPreparing data for {ticker}")
            
            # Create new DataFrame for processed data
            processed_data = pd.DataFrame(index=data.index)
            
            # Extract price data based on column structure
            if isinstance(data.columns, pd.MultiIndex):
                for price_type in ['Close', 'Open', 'High', 'Low', 'Volume']:
                    if (price_type, ticker) in data.columns:
                        processed_data[price_type] = data[(price_type, ticker)]
            else:
                processed_data = data[['Close', 'Open', 'High', 'Low', 'Volume']].copy()
            
            # Calculate technical indicators
            try:
                processed_data['SMA_20'] = processed_data['Close'].rolling(window=20).mean()
                processed_data['EMA_20'] = processed_data['Close'].ewm(span=20, adjust=False).mean()
                
                # Calculate RSI
                delta = processed_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                processed_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Add momentum indicators
                processed_data['ROC'] = processed_data['Close'].pct_change(periods=20)
            except Exception as e:
                print(f"Error calculating technical indicators: {e}")
            
            # Clean any NaN values
            processed_data = processed_data.dropna()
            
            print(f"Processed data shape: {processed_data.shape}")
            return processed_data
            
        except Exception as e:
            print(f"Error preparing data for {ticker}: {str(e)}")
            return pd.DataFrame()
   



    def validate_data(self, ticker, data):
        """Validate data with enhanced checks"""
        try:
            if data is None or data.empty:
                logger.error(f"Empty data received for {ticker}")
                return False
                
            min_required_points = 30
            required_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            
            # Handle both MultiIndex and regular columns
            if isinstance(data.columns, pd.MultiIndex):
                ticker_columns = [(col, ticker) for col in required_columns]
                data_subset = data[ticker_columns] if all(col in data.columns for col in ticker_columns) else None
            else:
                data_subset = data[required_columns] if all(col in data.columns for col in required_columns) else None
                
            if data_subset is None or data_subset.empty:
                logger.error(f"Missing required columns for {ticker}")
                return False
                
            if len(data_subset) < min_required_points:
                logger.error(f"Insufficient data points for {ticker}: {len(data_subset)} < {min_required_points}")
                return False
                
            # Check for excessive missing values
            missing_pct = data_subset.isnull().mean()
            if any(missing_pct > 0.1):  # More than 10% missing
                logger.error(f"Too many missing values for {ticker}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating data for {ticker}: {e}")
            return False





    def normalize_fundamental_score(self, fundamental_data, is_etf):
        """Improved fundamental score normalization"""
        try:
            if not fundamental_data:
                return 0, 0
                
            # Define expected ranges for each metric
            metric_ranges = {
                'P/E': {'min': 0, 'max': 50, 'weight': 1.5},
                'EPS': {'min': -10, 'max': 100, 'weight': 2.0},
                'Debt_Ratio': {'min': 0, 'max': 200, 'weight': -1.0},  # Higher debt is negative
                'ROE': {'min': -50, 'max': 50, 'weight': 2.0},
                'Current_Ratio': {'min': 0, 'max': 3, 'weight': 1.0},
                'Gross_Margin': {'min': 0, 'max': 100, 'weight': 1.5},
                'Operating_Margin': {'min': -50, 'max': 50, 'weight': 1.5},
                'Revenue_Growth': {'min': -50, 'max': 100, 'weight': 2.0},
                'Profit_Margin': {'min': -50, 'max': 50, 'weight': 2.0}
            }
            
            # Calculate normalized scores for each metric
            scores = []
            weights = []
            for metric, value in fundamental_data.items():
                if metric in metric_ranges and value is not None:
                    range_info = metric_ranges[metric]
                    # Normalize to [-1, 1] range
                    normalized = (value - range_info['min']) / (range_info['max'] - range_info['min'])
                    normalized = max(-1, min(1, normalized * 2 - 1))  # Scale to [-1, 1]
                    scores.append(normalized * range_info['weight'])
                    weights.append(abs(range_info['weight']))
            
            if not scores:
                return 0, 0
                
            # Calculate weighted average
            total_score = sum(scores)
            valid_metrics = len(scores)
            
            # Normalize final score to [-1, 1] range
            if total_score == 0:
                return 0, valid_metrics
            
            normalized_score = total_score / sum(weights)
            return normalized_score, valid_metrics
            
        except Exception as e:
            print(f"Error in normalize_fundamental_score: {e}")
            return 0, 0
            
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
        """Generate recommendations with improved debugging and error handling"""
        recommendations = {}
        detailed_analysis = {}
        
        print("\nStarting recommendation generation for tickers:", tickers)
        
        try:
            # Download data for all tickers at once
            print("Downloading data for all tickers...")
            data = yf.download(tickers, period="1y", interval="1d", progress=False)
            print(f"Downloaded data shape: {data.shape}")
            
            # Process each ticker only once
            processed_tickers = set()
            
            for ticker in tickers:
                if ticker in processed_tickers:
                    continue
                    
                try:
                    print(f"\n{'='*50}")
                    print(f"Processing {ticker}...")
                    
                    # Validate data
                    if not self.validate_data(ticker, data):
                        print(f"Validation failed for {ticker}")
                        continue
                    
                    # Prepare data
                    prepared_data = self.prepare_training_data(ticker, data)
                    if prepared_data.empty:
                        print(f"Failed to prepare data for {ticker}")
                        continue
                        
                    print(f"Prepared data shape: {prepared_data.shape}")
                    
                    # Get current price
                    if isinstance(data.columns, pd.MultiIndex):
                        current_price = float(data[('Close', ticker)].iloc[-1])
                    else:
                        current_price = float(data['Close'].iloc[-1])
                    print(f"Current price for {ticker}: ${current_price:.2f}")
                    
                    # Train model and get prediction
                    try:
                        print(f"Training model for {ticker}...")
                        model = self.train_prediction_model(prepared_data)
                        latest_data = prepared_data.iloc[-1:]
                        feature_cols = [col for col in prepared_data.columns if col != 'Close']
                        prediction = model.predict(latest_data[feature_cols])
                        expected_return = (prediction[0] - current_price) / current_price
                        print(f"Prediction: ${prediction[0]:.2f}")
                        print(f"Expected return: {expected_return:.2%}")
                    except Exception as e:
                        print(f"Error in prediction for {ticker}: {str(e)}")
                        continue
                    
                    # Get fundamental and sentiment scores
                    try:
                        fundamental_data = self.fetch_fundamental_data(ticker)
                        is_etf = self.is_etf(ticker)
                        normalized_fundamental_score, valid_metrics = self.normalize_fundamental_score(
                            fundamental_data, is_etf)
                        sentiment_data = self.fetch_news_sentiment(ticker)
                        sentiment_score = sentiment_data.get('avg_sentiment', 0)
                    except Exception as e:
                        print(f"Error in scoring for {ticker}: {str(e)}")
                        continue
                    
                    # Calculate final score
                    total_score = (
                        expected_return * 0.5 + 
                        normalized_fundamental_score * 0.3 + 
                        sentiment_score * 0.2
                    )
                    
                    # Store results
                    detailed_analysis[ticker] = {
                        'total_score': total_score,
                        'expected_return': expected_return,
                        'fundamental_score': normalized_fundamental_score,
                        'sentiment_score': sentiment_score,
                        'current_price': current_price,
                        'metrics_count': valid_metrics
                    }
                    
                    recommendations[ticker] = total_score
                    processed_tickers.add(ticker)
                    print(f"Successfully processed {ticker}")
                    
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
                    continue
                    
            # Print final summary
            print("\n" + "="*50)
            print(f"Successfully processed {len(recommendations)} stocks")
            if recommendations:
                print("\nTop recommendations:")
                for ticker, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]:
                    analysis = detailed_analysis[ticker]
                    print(f"{ticker}: Score={score:.2f}, Return={analysis['expected_return']:.2%}, "
                        f"Fundamental={analysis['fundamental_score']:.2f}, "
                        f"Sentiment={analysis['sentiment_score']:.2f}")
            else:
                print("No recommendations generated")
                
            return recommendations, detailed_analysis
            
        except Exception as e:
            print(f"Critical error in recommendation generation: {e}")
            return {}, {}






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
        """Enhanced commodity ETF hedging analysis with proper MultiIndex handling"""
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
            
            # Download commodity ETF data
            commodity_data = yf.download(list(commodity_etfs.keys()), period="6mo")
            
            # Handle both MultiIndex and regular DataFrame cases
            if isinstance(commodity_data.columns, pd.MultiIndex):
                # Extract Close prices from MultiIndex
                close_data = pd.DataFrame()
                for etf in commodity_etfs.keys():
                    if ('Close', etf) in commodity_data.columns:
                        close_data[etf] = commodity_data[('Close', etf)]
            else:
                close_data = commodity_data['Close']
                
            # Calculate returns and other metrics
            returns = close_data.pct_change().dropna()
            volatilities = returns.std() * np.sqrt(252)
            
            # Calculate momentum scores using the last 126 trading days (approximately 6 months)
            momentum = (close_data.iloc[-1] / close_data.iloc[-126] - 1)
            
            for etf in commodity_etfs:
                if etf in returns.columns:
                    # Calculate volatility score (lower volatility is better)
                    vol_score = 1 - (volatilities[etf] / volatilities.max())
                    
                    # Calculate momentum score (higher momentum is better)
                    mom_score = (momentum[etf] - momentum.min()) / (momentum.max() - momentum.min())
                    
                    # Combine scores
                    score = (vol_score + mom_score) / 2
                    
                    # Determine allocation based on market correlation
                    if score > 0.5:
                        if market_correlation > 0.7:
                            hedging_allocation[etf] = score * 0.05  # Higher allocation for high correlation
                        else:
                            hedging_allocation[etf] = score * 0.03  # Lower allocation for low correlation
            
            # Normalize allocations to respect maximum commodity exposure
            if hedging_allocation:
                total_allocation = sum(hedging_allocation.values())
                max_commodity_exposure = 0.20  # 20% maximum commodity exposure
                
                if total_allocation > 0:
                    hedging_allocation = {
                        k: (v/total_allocation) * max_commodity_exposure 
                        for k, v in hedging_allocation.items()
                    }
            
            return hedging_allocation
            
        except Exception as e:
            print(f"Error in analyze_commodity_hedging: {str(e)}")
            return {}
    





    def optimize_dynamic_portfolio(self, timeframe, tickers):
        """Optimize portfolio with proper risk handling"""
        try:
            market_conditions = self.analyze_market_conditions()
            recommendations, detailed_analysis = self.generate_enhanced_recommendations(tickers)
            
            # Calculate risk contributions as a dictionary
            risk_contributions = {}
            portfolio_vol = self.calculate_portfolio_volatility(tickers)
            
            # Calculate individual stock volatilities
            stock_data = yf.download(tickers, period="1y")['Close']
            for ticker in tickers:
                if isinstance(stock_data, pd.DataFrame):
                    returns = stock_data[ticker].pct_change().dropna()
                else:
                    returns = stock_data.pct_change().dropna()
                risk_contributions[ticker] = float(returns.std() * np.sqrt(252))
            
            # Dynamic allocation based on market conditions and risk
            allocations = self._calculate_dynamic_allocations(
                recommendations=recommendations,
                market_conditions=market_conditions,
                risk_contributions=risk_contributions
            )
            
            return {
                'market_conditions': market_conditions,
                'allocations': allocations,
                'risk_metrics': risk_contributions
            }
            
        except Exception as e:
            logger = logging.getLogger('StockAnalysis')
            logger.error(f"Error in portfolio optimization: {e}")
            return None
        


    


    def _calculate_dynamic_allocations(self, recommendations, market_conditions, risk_contributions):
        """Calculate allocations with improved risk management"""
        stress_level = market_conditions['market_stress']
        
        # Adjust base allocations for market stress
        equity_allocation = max(0.3, min(0.6, 0.6 * (1 - stress_level)))
        commodity_allocation = min(0.35, 0.25 + (stress_level * 0.2))
        defensive_allocation = min(0.35, 0.15 + (stress_level * 0.3))
        
        # Calculate position sizes with risk adjustment
        position_sizes = {}
        for ticker, score in recommendations.items():
            risk_adjustment = 1 - risk_contributions.get(ticker, 0.5)
            position_size = (
                equity_allocation * 
                (score / sum(recommendations.values())) * 
                risk_adjustment
            )
            position_sizes[ticker] = min(
                self.config['position_limits']['max'],
                max(self.config['position_limits']['min'], position_size)
            )
            
        return {
            'stocks': position_sizes,
            'commodities': self._allocate_commodities(commodity_allocation),
            'defensive': defensive_allocation
        }

    def _allocate_commodities(self, total_allocation):
        """Allocate commodities with proper error handling"""
        try:
            commodity_weights = {
                'GLD': 0.3,  # Gold
                'SLV': 0.2,  # Silver
                'DBC': 0.2,  # Broad commodities
                'USO': 0.15, # Oil
                'UNG': 0.15  # Natural gas
            }
            
            return {
                ticker: weight * total_allocation 
                for ticker, weight in commodity_weights.items()
            }
        except Exception as e:
            logger = logging.getLogger('StockAnalysis')
            logger.error(f"Error in commodity allocation: {e}")
            return {}


    
    def _calculate_market_stress(self, volatility, vix, trend_strength):
        """Calculate market stress with improved normalization"""
        try:
            # Normalize inputs
            vol_score = min(1.0, max(0.0, volatility / 0.4))
            vix_score = min(1.0, max(0.0, vix / 35.0))
            trend_score = min(1.0, max(0.0, -trend_strength / 0.1))
            
            # Weight and combine scores
            weights = {
                'volatility': 0.4,
                'vix': 0.4,
                'trend': 0.2
            }
            
            stress_score = (
                vol_score * weights['volatility'] +
                vix_score * weights['vix'] +
                trend_score * weights['trend']
            )
            
            return float(min(1.0, max(0.0, stress_score)))
            
        except Exception as e:
            logger = logging.getLogger('StockAnalysis')
            logger.error(f"Error calculating market stress: {e}")
            return 0.5

def fetch_all_tickers():
    """Fetch all tickers for analysis"""
    #islamic_us_etfs = ['SPUS', 'HLAL', 'SPSK', 'SPRE', 'SPTE', 'SPWO', 'UMMA']
    #top_tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'INTC', 'CSCO']
    #top_etfs = ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO', 'VTV', 'VUG', 'VOO']
    #commodity_etfs = ['GLD', 'SLV', 'USO', 'UNG', 'PPLT', 'PALL', 'WEAT', 'CORN', 'DBA', 'DBB', 'DBC', 'DBO', 'DBP']
    #sample_stocks = ['AAPL', 'TSLA', 'MSFT', 'AMZN', 'GOOGL', 'NVDA']
    
    #all_needed_tickers  = islamic_us_etfs+commodity_etfs+top_tech
    ## Return all categories of tickers
    #return all_needed_tickers

    """Fetch a focused list of tickers for analysis"""
    islamic_us_etfs = ['SPUS', 'HLAL']  # Core Islamic ETFs
    tech_stocks = ['GOOGL', 'AMZN', 'TSLA', 'AAPL', 'MSFT']  # Major tech stocks
    commodities = ['SLV', 'GLD', 'USO', 'DBC', 'UNG']  # Core commodities
    
    # Return deduplicated list
    return list(set(islamic_us_etfs + tech_stocks + commodities))

# Main function remains the same
def main():
    # Initialize API keys
    api_keys = {
        'finnhub': 'cu2ha6pr01qh0l7ha4mgcu2ha6pr01qh0l7ha4n0',
        'fred': 'cd852e18eff164cf69663b2b638f9d1e'
    }
    
    try:
        logger.info("\n====== AMAI Stock Advisor Analysis ======")
        logger.info("Initializing stock analyzer...")
        analyzer = EnhancedStockAnalysis(api_keys)
        
        # Get tickers
        tickers = fetch_all_tickers()
        logger.info(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}")
        
        # 1. Analyze market conditions
        logger.info("\n=== Market Conditions Analysis ===")
        market_conditions = analyzer.analyze_market_conditions()
        
        logger.info(f"Current market stress level: {market_conditions['market_stress']*100:.2f}%")
        logger.info(f"Current volatility: {market_conditions['current_volatility']*100:.2f}%")
        logger.info(f"VIX level: {market_conditions['vix_level']:.2f}")
        logger.info(f"Recent market trend: {market_conditions['market_trend']:.2f}%")
        
        # 2. Generate stock recommendations
        logger.info("\n=== Stock Recommendations ===")
        recommendations, detailed_analysis = analyzer.generate_enhanced_recommendations(tickers)
        
        # Sort and display top recommendations
        sorted_recommendations = sorted(detailed_analysis.items(), 
                                     key=lambda x: x[1]['total_score'], 
                                     reverse=True)
        
        logger.info("\nTop Stock Recommendations:")
        for ticker, analysis in sorted_recommendations[:5]:
            logger.info(f"\nStock: {ticker}")
            logger.info(f"Total Score: {analysis['total_score']:.2f}")
            logger.info(f"Expected Return: {analysis['expected_return']*100:.2f}%")
            logger.info(f"Fundamental Score: {analysis['fundamental_score']:.2f}")
            logger.info(f"Sentiment Score: {analysis['sentiment_score']:.2f}")
            logger.info(f"Current Price: ${analysis['current_price']:.2f}")
        
        # 3. Calculate portfolio metrics
        logger.info("\n=== Portfolio Analysis ===")
        portfolio_volatility = analyzer.calculate_portfolio_volatility(tickers)
        market_correlation = analyzer.calculate_market_correlation(tickers)
        
        logger.info(f"Portfolio Volatility: {portfolio_volatility*100:.2f}%")
        logger.info(f"Market Correlation: {market_correlation:.2f}")
        
        # 4. Analyze hedging opportunities
        logger.info("\n=== Hedging Recommendations ===")
        hedging_allocation = analyzer.analyze_commodity_hedging(
            portfolio_volatility, 
            market_correlation
        )
        
        if hedging_allocation:
            logger.info("\nRecommended Hedging Allocation:")
            for etf, allocation in hedging_allocation.items():
                logger.info(f"{etf}: {allocation*100:.1f}%")
        else:
            logger.info("No hedging allocation recommended at this time")
        
        # 5. Generate optimized portfolio
        logger.info("\n=== Optimized Portfolio Allocation ===")
        timeframes = ['1w', '1m', '3m', '1y']
        
        for timeframe in timeframes:
            logger.info(f"\nOptimized Portfolio for {timeframe} horizon:")
            portfolio = analyzer.optimize_dynamic_portfolio(timeframe, tickers)
            
            if portfolio:
                logger.info("\nStock Allocations:")
                for ticker, weight in portfolio['allocations']['stocks'].items():
                    logger.info(f"{ticker}: {weight*100:.1f}%")
                    
                logger.info("\nCommodity Allocations:")
                for ticker, weight in portfolio['allocations']['commodities'].items():
                    logger.info(f"{ticker}: {weight*100:.1f}%")
                    
                logger.info(f"\nDefensive Allocation: {portfolio['allocations']['defensive']*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        logger.error("Please check API keys and data source connectivity")

if __name__ == "__main__":
    main()