from math import log, e
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq

'''
    # USING Markov chain 
    def calculate_option_price(self, spot, strike, r, dte, sigma, is_call):
        d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * dte) / (sigma * np.sqrt(dte))
        # d1 = (np.log(spot / strike) + (r + sigma ** 2) * dte) / (sigma * np.sqrt(dte))

        d2 = d1 - sigma * np.sqrt(dte)

        if is_call:
            option_price = spot * norm.cdf(d1) - strike * np.exp(-r * dte) * norm.cdf(d2)
        else:
            option_price = strike * np.exp(-r * dte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return option_price
    def implied_volatility(self, args, option_price, is_call, tolerance=1e-6, max_iter=10000):
        iv = 0.5  # Initial guess for IV
        spot = float(args[0])
        strike = float(args[1])
        r = float(args[2] / 100)
        dte = float(args[3] / 365)
        for i in range(max_iter):
            option_price_calculated = self.calculate_option_price(spot, 
            strike, 
            r, dte, iv, 
            is_call)
            vega =( spot * np.sqrt(dte) * norm.pdf((np.log(spot / strike) + 
            (r + 0.5 * iv ** 2) * dte) / (iv * np.sqrt(dte))))
            diff = option_price_calculated - option_price
            print(option_price_calculated)
            if abs(diff) < tolerance:
                return iv
            iv -= diff / vega
        return iv
'''
class BlackScholes:
    def __init__(self, args, volatility = None, call_price = None, put_price = None):
        """
        Initializes BlackScholes class with the following arguments:

        args = [spot, strike, r, dte]

        spot: underlying asset price
        strike: strike price
        r: risk-free rate
        dte: time to maturity
        volatility: Volatility
        """
        
        self.underlying_price, self.strike_price, self.interest_rate, self.days_to_expiration = args
        self.volatility = volatility
        self.days_to_expiration = self.days_to_expiration / 365
        self.interest_rate = self.interest_rate/ 100
        self.args = args

        for i in ['call_price', 'put_price', 'call_delta', 'put_delta', 
                'call_delta2', 'put_delta2', 'call_theta', 'put_theta', 
                'call_rhod', 'put_rhod', 'call_rhof', 'call_rhof', 'vega', 
                'gamma', 'implied_volatility', 'put_call_parity']:
            self.__dict__[i] = None

        if not self.volatility:
            if call_price:
                self.call_price = call_price
                self.implied_volatility = self.implied_volatility_func(args = self.args, 
                                                                  option_price = self.call_price, 
                                                                  is_call = True)

            if put_price and not call_price:
                self.put_price = put_price
                self.implied_volatility = self.implied_volatility_func(args = self.args, 
                                                                  option_price = self.put_price, 
                                                                  is_call = False)

            if call_price and put_price:
                self.call_price = float(call_price)
                self.put_price = float(put_price)
                self.put_call_parity = self._parity()
        else:
            [self.call_price, self.put_price, self._a_, self._d1_, self._d2_] = self._price()
            [self.call_delta, self.put_delta] = self._delta()
            [self.call_delta2, self.put_delta2] = self._delta2()
            [self.call_theta, self.put_theta] = self._theta()
            [self.callRho, self.putRho] = self._rho()
            self.vega = self._vega()
            self.gamma = self._gamma()
            self.exerciceProbability = norm.cdf(self._d2_)
    
    # FFT
    def price_options_fft(self, spot, strike, dte, r, volatility_grid, is_call):
        """
        Implement FFT pricing for each volatility in the grid
        """
        d1 = ((np.log(spot / strike) + 
               (r + 0.5 * volatility_grid ** 2) * dte) / 
               (volatility_grid * np.sqrt(dte))
               )

        d2 = d1 - volatility_grid * np.sqrt(dte)

        # if option_type == 'call':
        if is_call:
            option_prices = (spot * norm.cdf(d1) - 
                             strike * np.exp(-r * dte) * norm.cdf(d2))
        # elif option_type == 'put':
        else:
            option_prices = strike * np.exp(-r * dte) * norm.cdf(-d2) - spot * norm.cdf(-d1)

        return option_prices

    # Function to calculate implied volatility from option prices using FFT
    def implied_volatility_func(self, args, option_price, is_call):
        """
        Calculates the implied volatility of an 
        option using the Black-Scholes model.

        Args:
            args (list): A list containing the following parameters:
                - spot (float): The current stock price.
                - strike (float): The strike price of the option.
                - r (float): The risk-free interest rate.
                - dte (float): The time to expiration in years.
            option_price (float): The price of the option.
            is_call (bool): Indicates whether the option is a call 
            option (True) or a put option (False).

        Returns:
            float: The implied volatility of the option.

        Raises:
            Exception: If the implied volatility cannot be calculated, 
            a default value of 1e-5 is returned.

        Note:
            This function uses the FFT method to price options for 
            different volatilities and then interpolates
            the option prices to find the implied volatility.
        """
        # Price options for the given volatility grid
        spot = float(args[0])
        strike = float(args[1])
        r = float(args[2] / 100)
        dte = float(args[3] / 365)
        volatility_grid = np.linspace(0.001, 5, 10000)
        # volatility_grid = np.linspace(0.01, 5, 100)
        option_prices = self.price_options_fft(spot,
                                               strike,
                                               dte,
                                               r,
                                               volatility_grid,
                                               is_call)
        # Interpolate option prices to find implied volatility
        interp_func = interp1d(option_prices, volatility_grid, kind='linear')
        # interp_func = interp1d(option_prices, volatility_grid, kind='next')
        # print('nearest')
        try:
            implied_volatility = interp_func(option_price)
            implied_volatility = float(implied_volatility) * 100
        except Exception as e:
            print('No IV', e)
            implied_volatility = 1e-5
        return implied_volatility
    
    def price_options(self, spot, strike, dte, r, sigma, is_call):
        # spot, strike, r, dte = args
        # r = float(r) / 100
        # dte = float(dte) / 365
        # sigma = float(sigma) / 100
        a = sigma * (dte ** 0.5)
        d1 = (np.log(spot / strike) + (r + (sigma**2) / 2) * dte) / a
        d2 = d1 - a
        if is_call:
            return spot * norm.cdf(d1) - strike * e ** (-r * dte) * norm.cdf(d2)
        else:
            return strike * e ** (-r * dte) * norm.cdf(-d2) - spot * norm.cdf(-d1)

    def implied_vol_call(self,  args, option_price, is_call):
        
        
        spot = float(args[0])
        strike = float(args[1])
        r = float(args[2] / 100)
        dte = float(args[3] / 365)
        # volatility_grid = np.linspace(0.001, 5, 10000)

        if dte <= 0:
            return np.nan

        def objective(sigma):
            return self.price_options(spot,
                                    strike,
                                    dte,
                                    r,
                                    sigma,
                                    is_call) - option_price

        try:
            implied_vol = brentq(objective, 1e-9, 5.0)
            return implied_vol
        except ValueError:
            return np.nan
        
    def _price(self):
        '''Returns the option price: [Call price, Put price]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = max(0.0, self.underlying_price - self.strike_price)
            put = max(0.0, self.strike_price - self.underlying_price)
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        call, put, a, d1, d2 = self.bs(self.args, self.volatility)
        return [call, put, a, d1, d2]

    def _delta(self):
        '''Returns the option delta: [Call delta, Put delta]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = 1.0 if self.underlying_price > self.strike_price else 0.0
            put = -1.0 if self.underlying_price < self.strike_price else 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        call = norm.cdf(self._d1_)
        put = -norm.cdf(-self._d1_)
        return [call, put]

    def _delta2(self):
        '''Returns the dual delta: [Call dual delta, Put dual delta]'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            call = -1.0 if self.underlying_price > self.strike_price else 0.0
            put = 1.0 if self.underlying_price < self.strike_price else 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        _b_ = e**-(self.interest_rate * self.days_to_expiration)
        call = -norm.cdf(self._d2_) * _b_
        put = norm.cdf(-self._d2_) * _b_
        return [call, put]

    def _vega(self):
        '''Returns the option vega'''
        if self.volatility == 0 or self.days_to_expiration == 0:
            return 0.0
        if self.strike_price == 0:
            raise ZeroDivisionError('The strike price cannot be zero')
        # print('vega 2')
        return (self.underlying_price * norm.pdf(self._d1_) * \
                (self.days_to_expiration ** 0.5)) / 100

    # def _theta(self):
    #     '''Returns the option theta: [Call theta, Put theta]'''
    #     _b_ = e**-(self.interest_rate * self.days_to_expiration)
    #     call = (-self.underlying_price * norm.pdf(self._d1_) * self.volatility / \
    #             (2 * (self.days_to_expiration**0.5) / 100)) - self.interest_rate * \
    #             self.strike_price * _b_ * norm.cdf(self._d2_)
    #     put = -self.underlying_price * norm.pdf(self._d1_) * self.volatility / \
    #             (2 * self.days_to_expiration**0.5) + self.interest_rate * \
    #             self.strike_price * _b_ * norm.cdf(-self._d2_)
    #     return [call / 365, put / 365]

    def _theta(self):
        '''Returns the option theta: [Call theta, Put theta]'''
        _b_ = e**(-(self.interest_rate * self.days_to_expiration))
        call = ((-self.underlying_price * norm.pdf(self._d1_) * self.volatility)/ \
                (2 * (self.days_to_expiration**0.5)) / 100) - (self.interest_rate * \
                self.strike_price * _b_ * norm.cdf(self._d2_))
        put = (-self.underlying_price * norm.pdf(self._d1_) * self.volatility / \
                (2 * (self.days_to_expiration**0.5)) / 100) + (self.interest_rate * \
                self.strike_price * _b_ * norm.cdf(-self._d2_))
        # print('theta 2')
        return [call / 365, put / 365]

    def _rho(self):
        '''Returns the option rho: [Call rho, Put rho]'''
        _b_ = e**-(self.interest_rate * self.days_to_expiration)
        call = self.strike_price * self.days_to_expiration * _b_ * \
                norm.cdf(self._d2_) / 100
        put = -self.strike_price * self.days_to_expiration * _b_ * \
                norm.cdf(-self._d2_) / 100
        return [call, put]

    def _gamma(self):
        '''Returns the option gamma'''
        return norm.pdf(self._d1_) / (self.underlying_price * self._a_)

    def _parity(self):
        '''Put-Call Parity'''
        return self.call_price - self.put_price - self.underlying_price + \
                (self.strike_price / \
                ((1 + self.interest_rate)**self.days_to_expiration))

    def bs(self, args, sigma):
        '''call put value'''
        spot, strike, r, dte = args
        r = float(r) / 100
        dte = float(dte) / 365
        sigma = float(sigma) / 100
        a = sigma * (dte ** 0.5)
        d1 = (log(spot / strike) + (r + (sigma**2) / 2) * dte) / a
        d2 = d1 - a
        call = spot * norm.cdf(d1) - strike * e ** (-r * dte) * norm.cdf(d2)
        put = strike * e ** (-r * dte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return call, put, a, d1, d2