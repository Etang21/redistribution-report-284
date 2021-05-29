"""
seller_one_price_solution.py

This code numerically computes the optimal seller price for a market designer to set
when they are restricted to one-price mechanisms, a fixed quantity Q and a fixed revenue R,
and when agents have non-linear preferences for money.

5.19.21: We start here with independent distributions of vM and vK.
"""

import numpy as np
from scipy.stats import uniform
from scipy.integrate import dblquad
from scipy.integrate import simps

import pandas as pd
import plotly.express as px

NUM_GOOD_TYPES = 10 # Number of discrete types for good valuations
NUM_MONEY_TYPES = 10 # Number of discrete types for money valuations

low_vK_sellers = 0
up_vK_sellers = 1

low_vM_sellers = 0.5
up_vM_sellers = 2.0

REVENUE = 1.0
QUANTITY = 0.2


def main():
    # Define sellers' valuation distributions for the good
    rv_vK_S = uniform(loc=low_vK_sellers, scale=(up_vK_sellers - low_vK_sellers))
    print("Mean of seller valuations for good: " + str(rv_vK_S.mean()))

    # Define sellers' valuation distributions for money
    rv_vM_S = uniform(loc=low_vM_sellers, scale=(up_vM_sellers - low_vM_sellers))
    print("Mean of seller valuations for money: " + str(rv_vM_S.mean()))

    # Test out discrete valuation function
    discrete_welfare = get_total_welfare_discretized(lambda xM, vM: exp_utility_for_money(xM, vM, 0.5),
                                    rv_vK_S, rv_vM_S, 2, REVENUE, QUANTITY)
    print("Discrete welfare: " + str(discrete_welfare))

    continuous_welfare = get_total_welfare(lambda xM, vM: exp_utility_for_money(xM, vM, 0.5),
                                    rv_vK_S, rv_vM_S, 2, REVENUE, QUANTITY)
    print("Continuous welfare: " + str(continuous_welfare))

    # Plot total welfare for various prices
    prices = [0.1 * i for i in range(0, 25)]
    plot_total_welfares(rv_vK_S, rv_vM_S, prices, 0.5)

    # Define constraints

    # Optimize


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
    df['Exponential Welfare'] = [get_total_welfare(lambda xM, vM: exp_utility_for_money(xM, vM, k), 
                                            rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY) for p_s in prices]
    df['Linear Welfare'] = [get_total_welfare(linear_utility_for_money, rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY) for p_s in prices]

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

def get_total_welfare_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q, low_vK_sellers = 0,
                                        up_vK_sellers = 1,
                                        low_vM_sellers = 0.5,
                                        up_vM_sellers = 2.0
                                    ):
    """ Total expected welfare of agents for given distribution, price, revenue, quantity """
    # Create grids of valuations for good and for money 
    vK_grid_1d = np.linspace(low_vK_sellers, up_vK_sellers, 20)
    vK_grid_2d = vK_grid_1d[:, None]
    vM_grid = np.linspace(low_vM_sellers, up_vM_sellers, 20)
    print("Made grids")

    # Compute number of agents who want to sell, and check that enough want to sell
    tau = get_tau_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q, vK_grid_1d, vM_grid)
    if tau < Q:
        print("Not enough sellers willing to sell (tau < Q) at price " + str(p_s))
        return 0

    # Utility for agents who successfully sell
    sale_utilities_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                            get_seller_sale_utility(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                            rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid)
    sale_utilities = (Q / tau) * simps(simps(sale_utilities_grid, vM_grid), vK_grid_1d)

    # Utility for agents who want to sell but are rationed out
    failed_sale_utilities_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                                    get_seller_no_sale_utility(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                                    rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid)
    failed_sale_utilities = ((tau - Q) / tau) * simps(simps(failed_sale_utilities_grid, vM_grid), vK_grid_1d)
    
    # Utility for agents who do not want to sell
    no_sale_utilities_grid = (~will_seller_sell(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q)) * \
                            get_seller_no_sale_utility(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                            rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid)
    no_sale_utilities = simps(simps(no_sale_utilities_grid, vM_grid), vK_grid_1d)

    # Sum total agent utilities
    return sale_utilities + failed_sale_utilities + no_sale_utilities

def get_tau_discretized(utility_for_money, rv_vK_S, rv_vM_S, p_s, R, Q, vK_grid_1d, vM_grid):
    """
    Computes tau, the proportion of sellers who are willing to sell.
    """
    vK_grid_2d = vK_grid_1d[:, None]
    will_sell_grid = will_seller_sell(utility_for_money, vK_grid_2d, vM_grid, p_s, R, Q) * \
                        rv_vK_S.pdf(vK_grid_2d) * rv_vM_S.pdf(vM_grid),
    return simps(simps(will_sell_grid, vM_grid), vK_grid_1d)

""" Functions to numerically compute total welfare for agents at a given price, revenue, and quantity """

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

def will_seller_sell(utility_for_money, vK, vM, p_s, R, Q):
    """
    Returns true if the seller will sell, false otherwise.
    vK, vM: Marginal values for the good and money.
    p_s, R, Q: price, total revenue, and quantity purchased.

    Assumes unit mass of sellers when computing lump-sum transfer R - p_s * Q.
    """
    return get_seller_sale_utility(utility_for_money, vK, vM, p_s, R, Q) >= \
            get_seller_no_sale_utility(utility_for_money, vK, vM, p_s, R, Q)


""" Functions to compute utility for individual agents """


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

if __name__ == '__main__':
    main()