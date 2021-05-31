"""
seller_one_price_solution.py

This code numerically computes the optimal seller price for a market designer to set
when they are restricted to one-price mechanisms, a fixed quantity Q and a fixed revenue R,
and when agents have non-linear preferences for money.

5.19.21: We start here with independent distributions of vM and vK.
"""

import numpy as np
import pandas as pd

from scipy.stats import uniform
from scipy.integrate import dblquad
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
import plotly.express as px

# Number of discrete types for good and money valuations
NUM_GOOD_TYPES = 100 
NUM_MONEY_TYPES = 100

# Lower and upper bound on valuations for good
low_vK_sellers = 0
up_vK_sellers = 1

# Lower and upper bound on valuations for money
low_vM_sellers = 0.5
up_vM_sellers = 2.0

# Desired revenue and quantity
REVENUE = 1.0
QUANTITY = 0.2


def main():
    # Define sellers' valuation distributions for the good
    rv_vK_S = uniform(loc=low_vK_sellers, scale=(up_vK_sellers - low_vK_sellers))
    print("Mean of seller valuations for good: " + str(rv_vK_S.mean()))

    # Define sellers' valuation distributions for money
    rv_vM_S = uniform(loc=low_vM_sellers, scale=(up_vM_sellers - low_vM_sellers))
    print("Mean of seller valuations for money: " + str(rv_vM_S.mean()))

    # Get welfare-maximizing prices for each utility function
    print(get_welfare_maximizing_price(linear_utility_for_money, rv_vK_S, rv_vM_S))
    print(get_competitive_price(linear_utility_for_money, rv_vK_S, rv_vM_S))

    print(get_welfare_maximizing_price(lambda xM, vM: exp_utility_for_money(xM, vM, 0.5), rv_vK_S, rv_vM_S))
    print(get_competitive_price(lambda xM, vM: exp_utility_for_money(xM, vM, 0.5), rv_vK_S, rv_vM_S))

    # Plot total welfare for various prices
    prices = [0.1 * i for i in range(0, 25)]
    plot_total_welfares(rv_vK_S, rv_vM_S, prices, 0.5)

    # Optimize


def get_welfare_maximizing_price(utility_for_money, rv_vK_S, rv_vM_S):
    """ 
    Return welfare-maximizing price for given distribution and utility for money.
    Returns (price, welfare) of this price.
    """
    res = minimize_scalar(lambda p_s: -get_total_welfare_discretized(utility_for_money, rv_vK_S, rv_vM_S, 
                                                                      p_s, REVENUE, QUANTITY))
    return res.x, res.fun

def get_competitive_price(utility_for_money, rv_vK_S, rv_vM_S):
    """ 
    Get competitive price for given specification.
    Returns (price, |Q - sold|) for specification.
    """
    tau_quantity_difference = lambda p_s: abs(QUANTITY - 
                                              get_tau_discretized(utility_for_money, 
                                                                    rv_vK_S, rv_vM_S, p_s,
                                                                    REVENUE, QUANTITY))
    res = minimize_scalar(tau_quantity_difference)
    return res.x, res.fun

""" Function to plot total welfares """

def plot_total_welfares(rv_vK_S, rv_vM_S, prices, k):
    """ 
    Plots total welfare for a series of prices.
    Plots both utility functions. 
    """
    # Set up pandas plotting
    pd.options.plotting.backend = "plotly" 

    # Compute total welfares
    df = pd.DataFrame()
    df['Price'] = prices
    df['Exponential Welfare'] = [get_total_welfare_discretized(lambda xM, vM: exp_utility_for_money(xM, vM, k), 
                                                                rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY) 
                                for p_s in prices]
    df['Linear Welfare'] = [get_total_welfare_discretized(linear_utility_for_money, rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY) 
                                for p_s in prices]

    # Plot welfares on chart
    fig = px.scatter(df, x='Price', y=['Linear Welfare', 'Exponential Welfare'],
                        title="Price vs. Total Seller Welfare: seller-side single-price mechanisms",
                        labels = {
                            'value': 'Total Seller Welfare'
                        }
                        )

    fig.add_annotation(text=F"R = {REVENUE}. Q = {QUANTITY}. vK ~ U(0, 1). vM ~ U(0.5, 2.0). <br>" + 
                            F"Exp. utility for money: w(xM; vM) = (xM * vM + 1)^{k}",
                        xref="paper", yref="paper",
                        x=1.0, y=-0.1, showarrow=False)
     
    fig.show()


""" Functions to dicretely compute total welfare for agents at a given price, revenue, and quantity """

def get_total_welfare_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q):
    """ Total expected welfare of agents for given distribution, price, revenue, quantity """
    
    # Make grids of good and money valuations
    vK_grid_1d = np.linspace(low_vK_sellers, up_vK_sellers, NUM_GOOD_TYPES)
    vM_grid_1d = np.linspace(low_vM_sellers, up_vM_sellers, NUM_MONEY_TYPES)
    vK_grid_2d = vK_grid_1d[:, None]

    # Compute number of agents who want to sell, and check that enough want to sell
    tau = get_tau_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q)
    if tau < Q:
        print("Not enough sellers willing to sell (tau < Q) at price " + str(p_s))
        return 0

    # Utility for agents who successfully sell
    sale_utilities_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                            get_seller_sale_utility(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                            rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid_1d)
    sale_utilities = (Q / tau) * simps(simps(sale_utilities_grid, vM_grid_1d), vK_grid_1d).item()

    # Utility for agents who want to sell but are rationed out
    failed_sale_utilities_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                                    get_seller_no_sale_utility(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                                    rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid_1d)
    failed_sale_utilities = ((tau - Q) / tau) * simps(simps(failed_sale_utilities_grid, vM_grid_1d), vK_grid_1d).item()
    
    # Utility for agents who do not want to sell
    no_sale_utilities_grid = (~will_seller_sell(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q)) * \
                            get_seller_no_sale_utility(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                            rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid_1d)
    no_sale_utilities = simps(simps(no_sale_utilities_grid, vM_grid_1d), vK_grid_1d).item()

    # Sum total agent utilities
    return sale_utilities + failed_sale_utilities + no_sale_utilities


def get_tau_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q):
    """
    Computes tau, the proportion of sellers who are willing to sell.
    """
    # Make grids of good and money valuations
    vK_grid_1d = np.linspace(low_vK_sellers, up_vK_sellers, NUM_GOOD_TYPES)
    vM_grid_1d = np.linspace(low_vM_sellers, up_vM_sellers, NUM_MONEY_TYPES)
    vK_grid_2d = vK_grid_1d[:, None]

    will_sell_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid_1d, p_s, R, Q) * \
                        rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid_1d),
    return simps(simps(will_sell_grid, vM_grid_1d), vK_grid_1d).item()


""" Functions to compute utility for individual agents """

def will_seller_sell(utility_for_money, vK, vM, p_s, R, Q):
    """
    Returns true if the seller will sell, false otherwise.
    vK, vM: Marginal values for the good and money.
    p_s, R, Q: price, total revenue, and quantity purchased.

    Assumes unit mass of sellers when computing lump-sum transfer R - p_s * Q.
    """
    return get_seller_sale_utility(utility_for_money, vK, vM, p_s, R, Q) >= \
            get_seller_no_sale_utility(utility_for_money, vK, vM, p_s, R, Q)

def get_seller_sale_utility(utility_for_money, vK, vM, p_s, R, Q):
    """ Returns utility of a seller upon selling their good. """
    return utility_for_money(p_s + R - p_s * Q, vM)

def get_seller_no_sale_utility(utility_for_money, vK, vM, p_s, R, Q):
    """ Returns utility of a seller if they do not sell their good """
    return vK + utility_for_money(R - p_s * Q, vM)

def linear_utility_for_money(xM, vM):
    """ 
    Linear utility of agent for quantity xM of money, with initial marginal valuation vM.
    Specification in originial Dworczak et al. (2020) paper.
    """
    return xM * vM

def exp_utility_for_money(xM, vM, k):
    """
    Exponential utility of agent for quantity xM of money, with initial marginal valuation vM.
    Satisfies three criteria for w laid out in paper.
    k: parameter in (0, 1) which determines the concavity of utility function.
    """
    return (xM * vM + 1) ** k


"""
Deprecated: Functions to more-exactly compute tau and total welfare using numerical integration.
Functions to numerically compute total welfare for agents at a given price, revenue, and quantity
"""

def get_total_welfare(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q):
    """ Total expected welfare of agents for given distribution, price, revenue, quantity """
    tau = get_tau(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q)

    # Check that enough agents want to sell
    if tau < Q:
        print("Not enough sellers willing to sell (tau < Q) at price " + str(p_s))
        return 0

    # Utility for agents who sell
    sale_utilities = (Q / tau) * dblquad(lambda vK, vM: int(will_seller_sell(utility_for_money, vK, vM, p_s, R, Q)) * 
                                        get_seller_sale_utility(utility_for_money, vK, vM, p_s, R, Q) *
                                        rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                                        low_vM_sellers, up_vM_sellers,
                                        lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                                        epsabs=0.01)[0]

    # Utility for agents who want to sell but are rationed out
    failed_sale_utilities = ((tau - Q) / tau) * dblquad(
                                    lambda vK, vM: int(will_seller_sell(utility_for_money, vK, vM, p_s, R, Q)) * 
                                    get_seller_no_sale_utility(utility_for_money, vK, vM, p_s, R, Q) *
                                    rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                                    low_vM_sellers, up_vM_sellers,
                                    lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                                    epsabs=0.01
                                )[0]
    
    # Utility for agents who do not want to sell
    no_sale_utilities = dblquad(
                            lambda vK, vM: int(not will_seller_sell(utility_for_money, vK, vM, p_s, R, Q)) * 
                            get_seller_no_sale_utility(utility_for_money, vK, vM, p_s, R, Q) *
                            rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                            low_vM_sellers, up_vM_sellers,
                            lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                            epsabs=0.01
                        )[0]

    # Sum total agent utilities
    return sale_utilities + failed_sale_utilities + no_sale_utilities

def get_tau(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q):
    """
    Computes tau, the proportion of sellers who are willing to sell.
    """
    result, abserr =  dblquad(lambda vK, vM: int(will_seller_sell(utility_for_money, vK, vM, p_s, R, Q)) * 
                              rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                              low_vM_sellers, up_vM_sellers,
                              lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                              epsabs=0.0001)
    return result


if __name__ == '__main__':
    main()