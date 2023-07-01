"""
Entropy,
conditional entropy,
information gain
"""
import math


"""
Assume we have a data frame with various columns
each column C represents a random variable X, and the values occuring in C represent the possible events/outcomes of X
p(X=x) would then be calculated based on how often the value x occurs in column C
"""

def Prob(df, X: str, x, conditions: dict = dict()):
    """
    computes P(X=x | conditions) using df, where condtions are of the form 'Variable': value, conditioning on Variable=value
    """

    # NOTE: can also just apply conditions to df directly, and not pass them through dict
    for var in conditions.keys():
        df = df[df[var] == conditions[var]]

    # count-based probability
    return len(df[df[X] == x]) / len(df)


def entropy(df, X: str):
    """
    computes H(X)
    """
    S = sum(list(map(lambda x: Prob(df, X, x) * math.log2(Prob(df, X, x)), df[X].unique())))
    return -S

def specific_cond_entropy(df, Y: str, X: str, x):
    """
    computes H(Y | X=x)
    """
    df = df[df[X] == x]
    return entropy(df, Y)

def expected_cond_entropy(df, Y: str, X: str):
    """
    computes H(Y | X)
    """
    f = lambda x: Prob(df, X, x) * specific_cond_entropy(df, Y, X, x)
    S = sum(list(map(f, df[X].unique())))
    return S

def information_gain(df, Y: str, X: str):
    """
    computes I(Y;X) = "how much information about Y do we gain by discovering the value of X?"
    """
    return entropy(df, Y) - expected_cond_entropy(df, Y, X)



