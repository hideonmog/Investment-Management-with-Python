import pandas as pd
import scipy.stats
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

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

def get_ind_size():
    '''
    '''
    ind = pd.read_csv('data/ind30_m_size.csv',
                        header = 0,
                        index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    '''
    '''
    ind = pd.read_csv('data/ind30_m_nfirms.csv',
                        header = 0,
                        index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_total_market_index_returns():
    '''
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    '''
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis = 'columns')
    ind_capweight = ind_mktcap.divide(total_mktcap, axis = 'rows')
    total_market_return = (ind_capweight * ind_return).sum(axis = 'columns')
    return total_market_return

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

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    '''
    Computes the annualised Sharpe Ratio of a set of returns
    '''
    # convert the annual risk free rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualise_rets(excess_ret, periods_per_year)
    ann_vol = annualise_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    ''''
    Weights -> Returns
    '''
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    '''
    Weights -> Vol
    '''
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style = '.-'):
    '''
    Plots the 2-asset efficient frontier
    '''
    if er.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2-asset frontiers')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
        })
    return ef.plot.line(x = 'Volatility', y = 'Returns', style = style)

def minimise_vol(target_return, er, cov):
    '''
    target_r -> W
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0), ) * n
    return_is_target = {
        'type': 'eq',
        'args': (er, ),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = scipy.optimize.minimize(portfolio_vol, init_guess,
                        args = (cov, ), method = 'SLSQP',
                        options = {'disp': False},
                        constraints = (return_is_target, weights_sum_to_1),
                        bounds = bounds
                        )
    return results.x

def optimal_weights(n_points, er, cov):
    '''
    -> list of weight sto run the optimiser on to minimise the vol
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimise_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(riskfree_rate, er, cov):
    '''
    Returns the weights of the portfolio that gives you the maximum sharpe ratio 
    given the riskfree rate, expected returns and a covariance matrix
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0), ) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Returns the negative of the sharpe ratio given weights
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    results = scipy.optimize.minimize(neg_sharpe_ratio, init_guess,
                        args = (riskfree_rate, er, cov, ), method = 'SLSQP',
                        options = {'disp': False},
                        constraints = (weights_sum_to_1),
                        bounds = bounds
                        )
    return results.x

def gmv(cov):
    '''
    Returns the weights of the Gloval Minimum Vol portfolio
    given the covariance matrix
    '''
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def plot_ef(n_points, er, cov, show_cml = False, style = '.-', riskfree_rate = 0, show_ew = False, show_gmv = False):
    '''
    Plots the N-asset efficient frontier
    '''
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
        })
    ax = ef.plot.line(x = 'Volatility', y = 'Returns', style = style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display equally weighted portfolio 
        ax.plot([vol_ew], [r_ew], color = 'goldenrod', marker = 'o', markersize = 10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # diplay global minimum variance portfolio
        ax.plot([vol_gmv], [r_gmv], color = 'firebrick', marker = 'o', markersize = 10)
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # add capital market line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', markersize = 10, linewidth = 2)

    return ax

def run_cppi(risky_r, safe_r = None, m = 3, start = 1000, floor = 0.8, riskfree_rate = 0.03, drawdown_constraint = None):
    '''
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History and Risky Weight History
    '''
    # set up cppi parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start * floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns = ['R'])
    
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12

    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown_constraint is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown_constraint)
        else:
            None
        cushion = (account_value - floor_value)/account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1 - risky_w
        risky_alloc = account_value * risky_w
        safe_alloc = account_value * safe_w
        # update account value for this time step
        account_value = (risky_alloc * (1 + risky_r.iloc[step])) + (safe_alloc * (1+ safe_r.iloc[step]))
        # save values to look at history
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth = start * (1 + risky_r).cumprod()
    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm': m,
        'start': start,
        'floor': floor,
        'risky_r': risky_r,
        'safe_R': safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate = 0.03):
    '''
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    '''
    ann_r = r.aggregate(annualise_rets, periods_per_year = 12)
    ann_vol = r.aggregate(annualise_vol, periods_per_year = 12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate = riskfree_rate, periods_per_year = 12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified = True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        'Annualised Return': ann_r,
        'Annualised Vol': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR (5%)': cf_var5,
        'Historic CVaR (5%)': hist_cvar5,
        'Sharpe Ratio': ann_sr,
        'Max Drawdown': dd
    })

def gbm(n_years = 10, n_scenarios = 1000, mu = 0.07, sigma = 0.15, steps_per_year = 12, s_0 = 100):
    '''
    Evolution of a Stock Price using a Geometric Brownian Motion Model
    '''
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)
    rets_plus_1 = np.random.normal(loc = 1 + mu * dt, scale = sigma * np.sqrt(dt), size = (n_steps, n_scenarios))
    # to prices
    prices = s_0 * pd.DataFrame((rets_plus_1)).cumprod()
    return prices

def gbm_updated(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of Geometric Brownian Motion trajectories, such as for Stock Prices through Monte Carlo
    :param n_years:  The number of years to generate data for
    :param n_paths: The number of scenarios/trajectories
    :param mu: Annualized Drift, e.g. Market Return
    :param sigma: Annualized Volatility
    :param steps_per_year: granularity of the simulation
    :param s_0: initial value
    :return: a numpy array of n_paths columns and n_years*steps_per_year rows
    """
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year) + 1
    # the standard way ...
    # rets_plus_1 = np.random.normal(loc=mu*dt+1, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    # without discretization error ...
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt, scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val

def show_gbm(n_scenarios, mu, sigma):
    '''
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    '''
    s_0 = 100
    prices = gbm_updated(n_scenarios = n_scenarios, mu = mu, sigma = sigma, s_0 = s_0)
    ax = prices.plot(legend = False, color = 'indianred', alpha = 0.5, linewidth = 2, figsize = (12, 5))
    ax.axhline(y = 100, ls = ':', color = 'black')
    ax.set_ylim(top = 400)
    # draw a dot at the origin
    ax.plot(0, s_0, marker = 'o', color = 'darkred', alpha = 0.2)

def show_cppi(n_scenarios = 50, mu = 0.07, sigma = 0.15, m = 3, floor = 0, riskfree_rate = 0.03, steps_per_year = 12, y_max = 100):
    '''
    Plot the results of a Monte Carlo Simulation of CPPI
    '''
    start = 100
    sim_rets = gbm_updated(n_scenarios = n_scenarios, mu = mu, sigma = sigma, prices = False, steps_per_year = steps_per_year)
    risky_r = pd.DataFrame(sim_rets)
    # run the back-test
    btr = run_cppi(risky_r = pd.DataFrame(sim_rets), riskfree_rate = riskfree_rate, m = m, start = start, floor = floor)
    wealth = btr['Wealth']
    # calculate terminal wealth stats
    y_max = wealth.values.max() * y_max/100
    terminal_wealth = wealth.iloc[-1]

    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failiure_mask = np.less(terminal_wealth, start * floor)
    n_failiures = failiure_mask.sum()
    p_fail = n_failiures/n_scenarios

    e_shortfall = np.dot(terminal_wealth - start * floor, failiure_mask)/n_failiures if n_failiures > 0 else 0.0

    # plot
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows = 1, ncols = 2, sharey = True, gridspec_kw = {'width_ratios':[3, 2]}, figsize = (24, 9))
    plt.subplots_adjust(wspace = 0.0)

    wealth.plot(ax = wealth_ax, legend = False, alpha = 0.3, color = 'indianred')
    wealth_ax.axhline(y = start, ls = ':', color = 'black')
    wealth_ax.axhline(y = start * floor, ls = '--', color = 'red')
    wealth_ax.set_ylim(top = y_max)

    terminal_wealth.plot.hist(ax = hist_ax, bins = 50, ec = 'w', fc = 'indianred', orientation = 'horizontal')
    hist_ax.axhline(y = start, ls = ':', color = 'black')
    hist_ax.axhline(y = tw_mean, ls = ':', color = 'blue')
    hist_ax.axhline(y = tw_median, ls = ':', color = 'purple')
    hist_ax.annotate(f'Mean: ${int(tw_mean)}', xy = (0.675, 0.9), xycoords = 'axes fraction', fontsize = 24)
    hist_ax.annotate(f'Median: ${int(tw_median)}', xy = (0.675, 0.85), xycoords = 'axes fraction', fontsize = 24)

    if (floor > 0.01):
        hist_ax.axhline(y = start * floor, ls = '--', color = 'red', linewidth = 3)
        hist_ax.annotate(f'Violations: {n_failiures} ({p_fail * 100: 2.2f}%)\n E(shortfall)=${e_shortfall: 2.2f}', xy = (0.675, 0.7),xycoords = 'axes fraction', fontsize = 24)

def discount(t, r):
    '''
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    '''
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(l, r):
    '''
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    '''
    dates = l.index
    discounts = discount(dates, r)
    return discounts.multiply(l, axis = 'rows').sum()

def funding_ratio(assets, liabilities, r):
    '''
    Computes the funding ratio of some assets given liabilities and interest rate
    '''
    return pv(assets, r)/pv(liabilities, r)

def inst_to_ann(r):
    ''' 
    Converts short rate to an annualised rate 
    '''
    return np.expm1(r)

def ann_to_inst(r):
    ''' 
    Convert annual rate to a short rate 
    '''
    return np.log1p(r)

def cir(n_years = 10, n_scenarios = 1, a = 0.05, b = 0.03, sigma = 0.05, steps_per_year = 12, r_0 = None):
    ''' 
    Generate random interest rate eveolution over time using the CIR model
    b and r_0 are assured to be the annualised rates, not the short rate
    and the returned values are the annualised rates as well
    '''
    if r_0 is None: r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year

    num_steps = int(n_years * steps_per_year) + 1
    shock = np.random.normal(0, scale = np.sqrt(dt), size = (num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0

    # for price generation
    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at time t as well ...
        prices[step] = price(n_years-step*dt, rates[step])

    rates = pd.DataFrame(data = inst_to_ann(rates), index=range(num_steps))
    # for prices
    prices = pd.DataFrame(data = prices, index = range(num_steps))
    return rates, prices

def bond_cash_flows(maturity, principal = 100, coupon_rate = 0.03, coupons_per_year = 12):
    '''
    Returns a series of cash flows generated by a bond
    indexed by coupon number
    '''
    n_coupons = round(maturity * coupons_per_year)
    coupon_amt = (principal * coupon_rate)/coupons_per_year
    coupon_times = np.arange(1, n_coupons + 1)
    cash_flows = pd.Series(data = coupon_amt, index = coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows

def bond_price(maturity, principal = 100, coupon_rate = 0.03, coupons_per_year = 12, discount_rate = 0.03):
    '''
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not desgined to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e The index of the discount_rate DataFrame is assumed to be the coupon number
    '''
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index = pricing_dates, columns = discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    '''
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g end of 3 months for quarterly div)
    and that dividends are reinvested in the bond
    '''
    coupons = pd.DataFrame(data = 0, index = monthly_prices.index, columns = monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype = int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def macaulay_duration(flows, discount_rate):
    '''
    Computes the Macaulay Duration of a sequence of cash flows
    '''
    discounted_flows = discount(flows.index, discount_rate) * flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights = weights)

def match_duration(cf_t, cf_s, cf_l, discount_rate):
    '''
    Returns the weight W in cf_s that, along with (1-W) in cf_l will have an effective
    duration that matches cf_t
    '''
    d_t = macaulay_duration(cf_t, discount_rate = discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate = discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate = discount_rate)
    return (d_l - d_t)/(d_l - d_s)