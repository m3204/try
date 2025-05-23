"""
Module for calculating option greeks.

This module provides the `BlackScholes` class for 
calculating option greeks using the Black-Scholes model.

:Example:

    >>> from opt_greeks import BlackScholes
    >>> args = [100, 100, 0.05, 1]
    >>> bs = BlackScholes(args, volatility=42.61890141059499)
    >>> bs.call_price
    0.8900000000189436

:Version: 0.0.1
:Author: Your Name
:Copyright: Copyright (c) 2022 Your Name
:License: MIT
"""


from .black_scholes import BlackScholes
__version__ = '1.1.2'
__all__ = ['BlackScholes']
