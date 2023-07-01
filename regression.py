"""
linear regression
generalized linear models
logistic regression
poisson regression
mixed effect models
"""



# Import the linear regression model class
from pymer4.models import Lm

# Import the lmm model class
from pymer4.models import Lmer

# Import Gaussian modeling
import statsmodels.formula.api as smf

"""

#Random intercepts only
(1 | Group)

#Random slopes only
(0 + Variable | Group)

#Random intercepts and slopes (and their correlation)
(Variable | Group)

#Random intercepts and slopes (without their correlation)
(1 | Group) + (0 + Variable | Group)

#Same as above but will not separate factors (see: https://rdrr.io/cran/lme4/man/expandDoubleVerts.html)
(Variable || Group)

#Random intercept and slope for more than one variable (and their correlations)
(Variable_1 + Variable_2 | Group)

Examples:
# Initialize model instance using 1 predictor with random intercepts and slopes
model = Lmer("passed ~ (1|category) + wa_num_subs_perc_correct", data=df, family='binomial') -> discrete output
model = Lmer("passed ~ (1|category) + wa_num_subs_perc_correct", data=df, family='gaussian') -> continuous output




random intercept for group
model = Lmer("quiz_grade ~ 1  + (1|group) + studying_hours", data=df_scaled, family='gaussian')

slope for group
model = Lmer("quiz_grade ~ 1  + (0 + studying_hours|group) ", data=df_scaled, family='gaussian')

random intercept and slope for group
model = Lmer("quiz_grade ~ (1 + studying_hours|group) ", data=df_scaled, family='gaussian')

random intercept and slope for groups AND interaction between the number of stuyding hours and time (weeks)
model = Lmer("quiz_grade ~  (1 + studying_hours*week|group) ", data=df_scaled, family='gaussian')


Fixed Effects: use categorical variable as a predictor in regression. 



Additive Factors Model (AFM)
model = Lmer("correct ~ (1|user_id) + (1|skill_name) + (0 + prev_attempts|skill_name)", data=X_train, family='binomial')


Performance Factors Analysis (PFA)
model = Lmer("correct ~ (1|user_id) + (1|skill_name) + (0 + before_correct_num|skill_name) + (0 + before_wrong_num|skill_name)", data=X_train, family='binomial')


"""

def linear_model(df, label, variables: list, family, intercept=True):
    variables_str = ' + '.join(variables)
    model = Lm(f"{label} ~ {'1' if intercept else '0'} + {variables_str}", data=df, family=family)
    print(model.fit())

    print("\n---------\nCoeffs:")
    print(model.coefs)
    return model

def mixed_effects_model(df, label, variables: list, random_effects_group_variable, random_slope_vars: list, random_intercepts: bool):
    # TODO: see examples
    pass


def logistic_regression():
    # TODO: sklearn.linear_model.LogisticRegression
    pass

