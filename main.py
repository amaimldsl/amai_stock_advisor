import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from textblob import TextBlob
import finnhub
from fredapi import Fred
import logging
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
        
        # Enhanced configuration with position limits
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
            },
            'position_limits': {
                'min': 0.02,  # Minimum position size (2%)
                'max': 0.15   # Maximum position size (15%)
            },
            'portfolio_constraints': {
                'max_equity': 0.60,    # Maximum equity allocation (60%)
                'max_commodity': 0.35,  # Maximum commodity allocation (35%)
                'min_defensive': 0.15   # Minimum defensive allocation (15%)
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
        """Fetch fundamental data with security type detection and appropriate metrics"""
        try:
            # First, determine security type
            yf_ticker = yf.Ticker(ticker)
            try:
                info = yf_ticker.info
                security_type = info.get('quoteType', '').upper()
            except:
                security_type = 'EQUITY'  # Default to equity if can't determine
                
            self.logger.info(f"\nFetching fundamental data for {ticker} (Type: {security_type})")
            
            if security_type in ['ETF', 'MUTUALFUND']:
                # ETF-specific metrics
                metrics = {
                    'AUM': self._safe_get_value(info, 'totalAssets'),
                    'Expense_Ratio': self._safe_get_value(info, 'expenseRatio'),
                    'NAV': self._safe_get_value(info, 'navPrice'),
                    'Dividend_Yield': self._safe_get_value(info, 'dividendYield'),
                    'Beta': self._safe_get_value(info, 'beta3Year'),
                    'YTD_Return': self._safe_get_value(info, 'ytdReturn'),
                    'Volume': self._safe_get_value(info, 'averageVolume')
                }
                
            elif security_type == 'COMMODITY' or ticker in ['GLD', 'SLV', 'USO', 'UNG', 'DBC']:
                # Commodity-specific metrics
                metrics = {
                    'Beta': self._safe_get_value(info, 'beta'),
                    'Volume': self._safe_get_value(info, 'averageVolume'),
                    'NAV': self._safe_get_value(info, 'navPrice'),
                    'Premium_Discount': self._calculate_premium_discount(info),
                    'Storage_Cost': 0.004,  # Typical storage cost for commodity ETFs
                    'Volatility': self._calculate_volatility(ticker)
                }
                
            else:
                # Regular stock metrics (using existing logic)
                financials = self.finnhub_client.company_basic_financials(ticker, 'all')
                
                metrics = {}
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
                
                for metric_name, sources in metric_mappings.items():
                    try:
                        if 'finnhub' in sources and financials and 'metric' in financials:
                            value = financials['metric'].get(sources['finnhub'])
                            if value is not None:
                                metrics[metric_name] = float(value)
                                continue
                                
                        if 'yfinance' in sources and sources['yfinance'] in info:
                            value = info[sources['yfinance']]
                            if value is not None:
                                metrics[metric_name] = float(value)
                                continue
                                
                        if 'fallback' in sources and sources['fallback'] in info:
                            value = info[sources['fallback']]
                            if value is not None:
                                metrics[metric_name] = float(value)
                                continue
                    except:
                        continue
            
            # Log summary of available metrics
            self.logger.info(f"\nMetrics found for {ticker}:")
            self.logger.info(f"Found {len(metrics)} metrics")
            self.logger.info("Available metrics: " + ", ".join(metrics.keys()))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {ticker}: {e}")
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





    def normalize_fundamental_score(self, fundamental_data, security_type):
        """Normalize fundamental scores based on security type"""
        try:
            if not fundamental_data:
                return 0, 0
                
            # Define metric ranges and weights based on security type
            if security_type in ['ETF', 'MUTUALFUND']:
                metric_ranges = {
                    'AUM': {'min': 1e6, 'max': 1e12, 'weight': 1.5},
                    'Expense_Ratio': {'min': 0, 'max': 0.01, 'weight': -2.0},
                    'Dividend_Yield': {'min': 0, 'max': 0.1, 'weight': 1.0},
                    'Beta': {'min': 0, 'max': 2, 'weight': -0.5},
                    'YTD_Return': {'min': -0.3, 'max': 0.3, 'weight': 1.0}
                }
            elif security_type == 'COMMODITY':
                metric_ranges = {
                    'Beta': {'min': 0, 'max': 2, 'weight': -1.0},
                    'Premium_Discount': {'min': -0.05, 'max': 0.05, 'weight': -1.0},
                    'Volatility': {'min': 0, 'max': 0.5, 'weight': -1.0}
                }
            else:
                # Original stock metrics (unchanged)
                metric_ranges = {
                    'P/E': {'min': 0, 'max': 50, 'weight': 1.5},
                    'EPS': {'min': -10, 'max': 100, 'weight': 2.0},
                    'Debt_Ratio': {'min': 0, 'max': 200, 'weight': -1.0},
                    'ROE': {'min': -50, 'max': 50, 'weight': 2.0},
                    'Current_Ratio': {'min': 0, 'max': 3, 'weight': 1.0},
                    'Gross_Margin': {'min': 0, 'max': 100, 'weight': 1.5},
                    'Operating_Margin': {'min': -50, 'max': 50, 'weight': 1.5},
                    'Revenue_Growth': {'min': -50, 'max': 100, 'weight': 2.0},
                    'Profit_Margin': {'min': -50, 'max': 50, 'weight': 2.0}
                }
                
            scores = []
            weights = []
            for metric, value in fundamental_data.items():
                if metric in metric_ranges and value is not None:
                    range_info = metric_ranges[metric]
                    normalized = (value - range_info['min']) / (range_info['max'] - range_info['min'])
                    normalized = max(-1, min(1, normalized * 2 - 1))
                    scores.append(normalized * range_info['weight'])
                    weights.append(abs(range_info['weight']))
            
            if not scores:
                return 0, 0
                
            total_score = sum(scores)
            valid_metrics = len(scores)
            normalized_score = total_score / sum(weights) if weights else 0
            
            return normalized_score, valid_metrics
            
        except Exception as e:
            self.logger.error(f"Error in normalize_fundamental_score: {e}")
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
    



    def optimize_dynamic_portfolio(self, timeframe, tickers, cached_recommendations=None):
        """Optimize portfolio with proper risk handling and caching"""
        try:
            market_conditions = self.analyze_market_conditions()
            
            # Use cached recommendations if provided, otherwise generate new ones
            if cached_recommendations is None:
                recommendations, detailed_analysis = self.generate_enhanced_recommendations(tickers)
            else:
                recommendations, detailed_analysis = cached_recommendations
            
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
        """Calculate allocations with improved risk management and position limits"""
        try:
            stress_level = market_conditions['market_stress']
            
            # Adjust base allocations for market stress within constraints
            equity_allocation = max(
                1 - self.config['portfolio_constraints']['max_commodity'] - self.config['portfolio_constraints']['min_defensive'],
                min(self.config['portfolio_constraints']['max_equity'], 
                    self.config['portfolio_constraints']['max_equity'] * (1 - stress_level))
            )
            
            commodity_allocation = min(
                self.config['portfolio_constraints']['max_commodity'],
                0.25 + (stress_level * 0.2)
            )
            
            defensive_allocation = max(
                self.config['portfolio_constraints']['min_defensive'],
                0.15 + (stress_level * 0.3)
            )
            
            # Calculate position sizes with risk adjustment
            position_sizes = {}
            total_score = sum(recommendations.values()) if recommendations else 1
            
            for ticker, score in recommendations.items():
                risk_adjustment = 1 - risk_contributions.get(ticker, 0.5)
                base_position = (equity_allocation * (score / total_score) * risk_adjustment)
                
                # Apply position limits
                position_sizes[ticker] = min(
                    self.config['position_limits']['max'],
                    max(self.config['position_limits']['min'], base_position)
                )
                
            # Normalize position sizes to ensure they sum to equity_allocation
            total_position = sum(position_sizes.values())
            if total_position > 0:
                position_sizes = {
                    ticker: (size / total_position) * equity_allocation 
                    for ticker, size in position_sizes.items()
                }
                
            return {
                'stocks': position_sizes,
                'commodities': self._allocate_commodities(commodity_allocation),
                'defensive': defensive_allocation
            }
            
        except Exception as e:
            self.logger.error(f"Error in dynamic allocation calculation: {str(e)}")
            return {
                'stocks': {},
                'commodities': {},
                'defensive': 0.15  # Default to minimum defensive allocation
            }
        
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


    def _allocate_commodities(self, total_allocation):
        """
        Allocate commodities with dynamic weights based on market conditions
        and proper error handling.
        
        Args:
            total_allocation (float): Total allocation for commodities (0-1)
            
        Returns:
            dict: Commodity ticker to allocation mapping
        """
        try:
            # Define base weights for different commodity types
            commodity_weights = {
                'GLD': {  # Gold
                    'base_weight': 0.30,
                    'stress_multiplier': 1.2  # Increase allocation during stress
                },
                'SLV': {  # Silver
                    'base_weight': 0.20,
                    'stress_multiplier': 1.1
                },
                'DBC': {  # Broad commodities
                    'base_weight': 0.20,
                    'stress_multiplier': 1.0
                },
                'USO': {  # Oil
                    'base_weight': 0.15,
                    'stress_multiplier': 0.9
                },
                'UNG': {  # Natural gas
                    'base_weight': 0.15,
                    'stress_multiplier': 0.9
                }
            }
            
            # Get current market conditions if available
            try:
                market_conditions = self.analyze_market_conditions()
                stress_level = market_conditions.get('market_stress', 0.5)
            except:
                stress_level = 0.5  # Default to moderate stress if analysis fails
                
            # Calculate dynamic weights based on market stress
            dynamic_weights = {}
            total_dynamic_weight = 0
            
            for ticker, props in commodity_weights.items():
                # Adjust weight based on market stress
                adjusted_weight = props['base_weight'] * (
                    1 + (props['stress_multiplier'] - 1) * stress_level
                )
                dynamic_weights[ticker] = adjusted_weight
                total_dynamic_weight += adjusted_weight
                
            # Normalize weights and apply total allocation
            allocations = {}
            if total_dynamic_weight > 0:  # Prevent division by zero
                for ticker, weight in dynamic_weights.items():
                    allocations[ticker] = (weight / total_dynamic_weight) * total_allocation
                    
            # Apply minimum and maximum constraints
            for ticker in allocations:
                allocations[ticker] = min(
                    self.config['position_limits']['max'],
                    max(self.config['position_limits']['min'], 
                        allocations[ticker])
                )
                
            # Log allocation details
            self.logger.info("Commodity Allocation Details:")
            for ticker, allocation in allocations.items():
                self.logger.info(f"{ticker}: {allocation*100:.1f}%")
                
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error in commodity allocation: {str(e)}")
            # Return safe default allocations in case of error
            default_allocation = total_allocation / 5  # Equal split among 5 commodities
            return {
                'GLD': default_allocation,
                'SLV': default_allocation,
                'DBC': default_allocation,
                'USO': default_allocation,
                'UNG': default_allocation
            }

    def _safe_get_value(self, info_dict, key):
        """Safely extract and convert value from info dictionary"""
        try:
            value = info_dict.get(key)
            return float(value) if value is not None else None
        except:
            return None

    def _calculate_premium_discount(self, info):
        """Calculate premium/discount to NAV"""
        try:
            if info.get('navPrice') and info.get('regularMarketPrice'):
                nav = float(info['navPrice'])
                price = float(info['regularMarketPrice'])
                return (price - nav) / nav
            return None
        except:
            return None

    def _calculate_volatility(self, ticker):
        """Calculate historical volatility"""
        try:
            data = yf.download(ticker, period="1y")['Close']
            returns = data.pct_change().dropna()
            return float(returns.std() * np.sqrt(252))  # Annualized volatility
        except:
            return None



class PortfolioRecommender:
    def __init__(self, stock_analyzer):
        self.analyzer = stock_analyzer
        self.timeframes = {
            '1w': {'days': 7, 'description': 'Weekly'},
            '1m': {'days': 30, 'description': 'Monthly'},
            '3m': {'days': 90, 'description': 'Quarterly'},
            '1y': {'days': 365, 'description': 'Yearly'}
        }
    
    def generate_recommendations(self, tickers):
        """Generate comprehensive trading recommendations across timeframes"""
        recommendations = {}
        
        # Get base analysis from the stock analyzer
        base_recommendations, detailed_analysis = self.analyzer.generate_enhanced_recommendations(tickers)
        market_conditions = self.analyzer.analyze_market_conditions()
        
        for timeframe, params in self.timeframes.items():
            # Optimize portfolio for each timeframe
            portfolio = self.analyzer.optimize_dynamic_portfolio(
                timeframe=timeframe,
                tickers=tickers,
                cached_recommendations=(base_recommendations, detailed_analysis)
            )
            
            # Sort stocks by expected return
            stocks_by_return = sorted(
                detailed_analysis.items(),
                key=lambda x: x[1]['expected_return'],
                reverse=True
            )
            
            # Get top 5 buys and sells
            top_buys = stocks_by_return[:5]
            top_sells = stocks_by_return[-5:]
            
            recommendations[timeframe] = {
                'portfolio_allocation': portfolio['allocations'],
                'market_conditions': market_conditions,
                'top_buys': [
                    {
                        'ticker': stock[0],
                        'expected_return': stock[1]['expected_return'] * 100,
                        'current_price': stock[1]['current_price'],
                        'target_price': stock[1]['current_price'] * (1 + stock[1]['expected_return'])
                    } for stock in top_buys
                ],
                'top_sells': [
                    {
                        'ticker': stock[0],
                        'expected_return': stock[1]['expected_return'] * 100,
                        'current_price': stock[1]['current_price'],
                        'target_price': stock[1]['current_price'] * (1 + stock[1]['expected_return'])
                    } for stock in top_sells
                ]
            }
        
        return recommendations

    def format_recommendations(self, recommendations):
        """Format recommendations into a clear, readable report"""
        report = []
        
        for timeframe, data in recommendations.items():
            period = self.timeframes[timeframe]['description']
            report.append(f"\n=== {period} Outlook ===")
            
            # Market conditions
            market_stress = data['market_conditions']['market_stress'] * 100
            report.append(f"\nMarket Stress Level: {market_stress:.1f}%")
            
            # Portfolio allocation
            report.append("\nRecommended Portfolio Allocation:")
            allocations = data['portfolio_allocation']
            report.append("Stocks:")
            for ticker, alloc in allocations['stocks'].items():
                report.append(f"  - {ticker}: {alloc*100:.1f}%")
            report.append("Commodities (Hedging):")
            for ticker, alloc in allocations['commodities'].items():
                report.append(f"  - {ticker}: {alloc*100:.1f}%")
            report.append(f"Defensive Assets: {allocations['defensive']*100:.1f}%")
            
            # Top buys
            report.append("\nTop 5 Stocks to Buy:")
            for stock in data['top_buys']:
                report.append(
                    f"  - {stock['ticker']}: Expected Gain = {stock['expected_return']:.1f}%"
                    f" (Current: ${stock['current_price']:.2f} → Target: ${stock['target_price']:.2f})"
                )
            
            # Top sells
            report.append("\nTop 5 Stocks to Sell:")
            for stock in data['top_sells']:
                report.append(
                    f"  - {stock['ticker']}: Expected Loss = {stock['expected_return']:.1f}%"
                    f" (Current: ${stock['current_price']:.2f} → Target: ${stock['target_price']:.2f})"
                )
        
        return "\n".join(report)

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

def main():
    # Initialize API keys
    api_keys = {
        'finnhub': 'cu2ha6pr01qh0l7ha4mgcu2ha6pr01qh0l7ha4n0',
        'fred': 'cd852e18eff164cf69663b2b638f9d1e'
    }
    
    # Initialize analyzers
    stock_analyzer = EnhancedStockAnalysis(api_keys)
    portfolio_recommender = PortfolioRecommender(stock_analyzer)
    
    # Get tickers for analysis
    tickers = fetch_all_tickers()
    
    # Generate recommendations
    recommendations = portfolio_recommender.generate_recommendations(tickers)
    
    # Print formatted recommendations
    print(portfolio_recommender.format_recommendations(recommendations))

if __name__ == "__main__":
    main()