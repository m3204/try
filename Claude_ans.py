import pandas as pd
import numpy as np

def analyze_iron_fly_patterns(df):
    """
    Analyzes iron fly trading data to identify potential directional patterns
    
    Parameters:
    df: pandas DataFrame with columns:
        - expiry: expiry date
        - entry_datetime: entry time of iron fly
        - spot_price: spot price at entry
        - atm_strike: ATM strike price
        - upper_range: upper boundary
        - lower_range: lower boundary
        - atm_call_iv: IV of ATM call
        - atm_put_iv: IV of ATM put
        - combined_premium: total premium collected
        - range_broken: 'upper' or 'lower' indicating which range was broken
    
    Returns:
    Dictionary containing pattern analysis results
    """
    results = {}
    
    # 1. Analyze IV Skew at Range Breaks
    df['iv_skew'] = df['atm_call_iv'] - df['atm_put_iv']
    results['iv_skew_patterns'] = {
        'upper_breaks_avg_skew': df[df['range_broken'] == 'upper']['iv_skew'].mean(),
        'lower_breaks_avg_skew': df[df['range_broken'] == 'lower']['iv_skew'].mean()
    }
    
    # 2. Premium to Spot Ratio Analysis
    df['premium_to_spot'] = df['combined_premium'] / df['spot_price'] * 100
    results['premium_patterns'] = {
        'high_premium_breaks': df[df['premium_to_spot'] > df['premium_to_spot'].mean()]['range_broken'].value_counts(),
        'low_premium_breaks': df[df['premium_to_spot'] <= df['premium_to_spot'].mean()]['range_broken'].value_counts()
    }
    
    # 3. Range Break Distance Analysis
    df['upper_distance'] = (df['upper_range'] - df['spot_price']) / df['spot_price'] * 100
    df['lower_distance'] = (df['spot_price'] - df['lower_range']) / df['spot_price'] * 100
    
    # 4. Consecutive Break Patterns
    df['prev_break'] = df['range_broken'].shift(1)
    results['consecutive_patterns'] = {
        'upper_after_upper': len(df[(df['range_broken'] == 'upper') & (df['prev_break'] == 'upper')]),
        'lower_after_lower': len(df[(df['range_broken'] == 'lower') & (df['prev_break'] == 'lower')]),
        'reversal_breaks': len(df[(df['range_broken'] != df['prev_break']) & (df['prev_break'].notna())])
    }
    
    # 5. Time-Based Pattern Analysis
    df['entry_hour'] = pd.to_datetime(df['entry_datetime']).dt.hour
    results['time_patterns'] = {
        'morning_breaks': df[df['entry_hour'] < 12]['range_broken'].value_counts(),
        'afternoon_breaks': df[df['entry_hour'] >= 12]['range_broken'].value_counts()
    }
    
    # 6. Volatility Impact Analysis
    df['avg_iv'] = (df['atm_call_iv'] + df['atm_put_iv']) / 2
    results['volatility_patterns'] = {
        'high_iv_breaks': df[df['avg_iv'] > df['avg_iv'].mean()]['range_broken'].value_counts(),
        'low_iv_breaks': df[df['avg_iv'] <= df['avg_iv'].mean()]['range_broken'].value_counts()
    }
    
    return results

def generate_trading_signals(df, results):
    """
    Generates directional trading signals based on identified patterns
    
    Parameters:
    df: Original DataFrame
    results: Pattern analysis results from analyze_iron_fly_patterns
    
    Returns:
    DataFrame with trading signals
    """
    signals = pd.DataFrame()
    
    # 1. IV Skew Based Signals
    signals['iv_skew_signal'] = np.where(
        df['iv_skew'] > df['iv_skew'].mean(),
        'LONG' if results['iv_skew_patterns']['upper_breaks_avg_skew'] > 0 else 'SHORT',
        'NEUTRAL'
    )
    
    # 2. Premium Based Signals
    signals['premium_signal'] = np.where(
        df['premium_to_spot'] > df['premium_to_spot'].mean(),
        'HIGH_PREMIUM',
        'LOW_PREMIUM'
    )
    
    # 3. Consecutive Break Signals
    signals['momentum_signal'] = np.where(
        (df['range_broken'] == df['prev_break']),
        df['range_broken'],
        'NEUTRAL'
    )
    
    # 4. Volatility Based Signals
    signals['volatility_signal'] = np.where(
        df['avg_iv'] > df['avg_iv'].mean(),
        'HIGH_VOL',
        'LOW_VOL'
    )
    
    return signals
