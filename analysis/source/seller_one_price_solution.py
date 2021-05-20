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

NUM_GOOD_TYPES = 10 # Number of discrete types for good valuations
NUM_MONEY_TYPES = 10 # Number of discrete types for money valuations

low_vK_sellers = 0
up_vK_sellers = 1

low_vM_sellers = 0
up_vM_sellers = 2

REVENUE = 1.0
QUANTITY = 0.2


def main():
    # Define sellers' valuation distributions for the good
    rv_vK_S = uniform(loc=low_vK_sellers, scale=(up_vK_sellers - low_vK_sellers))
    print("Mean of seller valuations for good: " + str(rv_vK_S.mean()))

    # Define sellers' valuation distributions for money
    rv_vM_S = uniform(loc=low_vM_sellers, scale=(up_vM_sellers - low_vM_sellers))
    print("Mean of seller valuations for money: " + str(rv_vM_S.mean()))


    # TODO (will move this to get objective), computes tau
    p_s = 0.9
    tau = get_tau(rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY)
    print("tau is: " + str(tau))

    # Define objective
    print("Total welfare objective for price " + str(p_s) + 
            " is: " + str(get_total_welfare(rv_vK_S, rv_vM_S, p_s, REVENUE, QUANTITY)))



    # Define constraints



    # Optimize


def utility_for_money(xM, vM):
    """ 
    Utility of agent for quantity xM of money, with initial marginal valuation vM.
    The functional form of this utility function can be changed according to constraints
    laid out in the paper.
    """
    return np.log(xM * vM + 1)
    # return xM * vM

def get_total_welfare(rv_vK_S, rv_vM_S, p_s, R, Q):
    """ Total expected welfare of agents for given distribution, price, revenue, quantity """
    tau = get_tau(rv_vK_S, rv_vM_S, p_s, R, Q)

    # Check that enough agents want to sell
    if tau < Q:
        return 0

    # Utility for agents who sell
    sale_utilities = (Q / tau) * dblquad(lambda vK, vM: int(will_seller_sell(vK, vM, p_s, R, Q)) * 
                                        get_seller_sale_utility(vK, vM, p_s, R, Q) *
                                        rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                                        low_vM_sellers, up_vM_sellers,
                                        lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                                        epsabs=0.0001)[0]

    # Utility for agents who want to sell but are rationed out
    failed_sale_utilities = ((tau - Q) / Q) * dblquad(
                                    lambda vK, vM: int(will_seller_sell(vK, vM, p_s, R, Q)) * 
                                    get_seller_no_sale_utility(vK, vM, p_s, R, Q) *
                                    rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                                    low_vM_sellers, up_vM_sellers,
                                    lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                                    epsabs=0.0001
                                )[0]
    
    # Utility for agents who do not want to sell
    no_sale_utilities = dblquad(
                            lambda vK, vM: int(not will_seller_sell(vK, vM, p_s, R, Q)) * 
                            get_seller_no_sale_utility(vK, vM, p_s, R, Q) *
                            rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                            low_vM_sellers, up_vM_sellers,
                            lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                            epsabs=0.0001
                        )[0]

    # Sum total agent utilities
    return sale_utilities + failed_sale_utilities + no_sale_utilities

def will_seller_sell(vK, vM, p_s, R, Q):
    """
    Returns true if the seller will sell, false otherwise.
    vK, vM: Marginal values for the good and money.
    p_s, R, Q: price, total revenue, and quantity purchased.

    Assumes unit mass of sellers when computing lump-sum transfer R - p_s * Q.
    """
    return get_seller_sale_utility(vK, vM, p_s, R, Q) >= get_seller_no_sale_utility(vK, vM, p_s, R, Q)


def get_tau(rv_vK_S, rv_vM_S, p_s, R, Q):
    """
    Computes tau, the proportion of sellers who are willing to sell.
    """
    result, abserr =  dblquad(lambda vK, vM: int(will_seller_sell(vK, vM, p_s, R, Q)) * 
                              rv_vK_S.pdf(vK) * rv_vM_S.pdf(vM),
                              low_vM_sellers, up_vM_sellers,
                              lambda vM: low_vK_sellers, lambda vM: up_vK_sellers,
                              epsabs=0.0001)
    return result


def get_seller_sale_utility(vK, vM, p_s, R, Q):
    """ Returns utility of a seller upon selling their good. """
    return utility_for_money(p_s + R - p_s * Q, vM)

def get_seller_no_sale_utility(vK, vM, p_s, R, Q):
    """ Returns utility of a seller if they do not sell their good """
    return vK + utility_for_money(R - p_s * Q, vM)

if __name__ == '__main__':
    main()


"""
Outdated code:

    # Discretize space of seller valuations for the good
    vK_S = np.linspace(low_vK_sellers, up_vK_sellers, num=NUM_GOOD_TYPES)
    f_vK_S = rv_vK_sellers.pdf(vK_S)
    F_vK_S = rv_vK_sellers.cdf(vK_S)
    # print("Discretized seller valuations for good: " + str(vK_S))
    # print("Discretized seller valuation CDFs for good: " + str(F_vK_S))

    # Discretize space of seller valuations for money
    vM_S = np.linspace(low_vM_sellers, up_vM_sellers, num=NUM_MONEY_TYPES)
    f_vM_S = rv_vM_sellers.pdf(vM_S)
    F_vM_S = rv_vM_sellers.cdf(vM_S)
    # print("Discretized seller valuations for money: " + str(vM_S))
    # print("Discretized seller valuation CDFs for money: " + str(F_vM_S))

"""