import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    '''
    wealth_index = 1000 * (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    '''
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    '''
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                        header = 0,
                        index_col = 0,
                        parse_dates = True,
                        na_values = -99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    '''
    Load and format the EDHEC Hedge Fund Index Returns
    '''
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv',
                        header = 0,
                        index_col = 0,
                        parse_dates = True,
                        infer_datetime_format = True)
    hfi = hfi/100
    hfi.index.name = None
    hfi.index = pd.to_datetime(hfi.index, format = '%Y%m').to_period('M')
    return hfi

def get_ind_returns():
    '''
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    '''
    ind = pd.read_csv('data/ind30_m_vw_rets.csv',
                        header = 0,
                        index_col = 0)/100
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def semideviation(r):
    '''
    Returns the semideviation (negative semideviation) of r
    r must be a Series or a DataFrame
    
    This is an approximation that is used for data with
    a mean close to zero (daily returns)
    '''
    is_negative = r < 0
    return r[is_negative].std(ddof = 0)

def semideviation3(r):
    '''
    Returns the semideviation (negative semideviation) of r
    r must be a Series or a DataFrame
    '''
    excess = r - r.mean()
    excess_negative = excess[excess < 0]
    excess_negative_square = excess_negative**2
    n_negative = (excess < 0).sum()
    return (excess_negative_square.sum()/n_negative)**0.5

def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof = 0
    sigma_r = r.std(ddof = 0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = 0.01):
    '''
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% confidence level by default 
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that 'level' percent of the returns
    fall below that number, and the (100-level) percent are above
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame') 

def var_gaussian(r, level = 5, modified = False):
    '''
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If 'modified' is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    '''
    # compute z score assuming gaussian
    z = scipy.stats.norm.ppf(level/100)
    if modified:
        # modify z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 - 3*z)*(k - 3)/24 +
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof = 0))

def cvar_historic(r, level = 5):
    '''
    Computes the Conditional historic VaR of Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else:
        raise TypeError('Expected r to be a Series or DataFrame')

def cvar_gaussian(r, level = 5, modified = False):
    '''
    Computes the Conditional parametric gaussian VaR of Series or DataFrame
    If 'modified' is True, then the modified CVaR is returned,
    using the Cornish-Fisher modification
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_gaussian(r, level = level, modified = modified)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level = level, modified = modified)
    else:
        raise TypeError('Expected r to be a Series or DataFrame')

def annualise_rets(r, periods_per_year):
    '''
    Annualises a set of returns
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

def annualise_vol(r, periods_per_year):
    '''
    Annualise the vol of a set of returns
    '''
    return r.std()*(periods_per_year**0.5)

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    '''
    Computes the annualised Sharpe Ratio of a set of returns
    '''
    # convert the annual risk free rate to per period
    rf_per_period = (1+risk_free_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualise_rets(excess_ret, periods_per_year)
    ann_vol = annualise_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol