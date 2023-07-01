"""
demographic parity
equalized odds
predictive value parity
disparate impact? -> lab

partial dependency plot PDP
LIME
"""


"""
df of following form:
    y_true: actual outcome
    y_pred: predicted outcome
    group: label for respective group
    *attributes: various columsn of attributes which each can take multiple values
"""

import pandas as pd

# conditions is list of (column, value) pairs
def cond_proba(df, y_col, y_val, conditions: list):
    for col, val in conditions:
        df = df[df[col] == val]

    return len(df[df[y_col] == y_val])/len(df)


def demographic_parity(df, A):
    parity = dict()
    bool_state = True
    for a in df[A].unique():
        pa = cond_proba(df, 'y_pred', 1, [(A, a)])
        parity[a] = pa
        for b in df[A].unique():
            pb = cond_proba(df, 'y_pred', 1, [(A, b)])
            bool_state = bool_state and pa == pb

    print(parity)
    return bool_state
            
            


# NOTE: A is the demographic group, T the true label
def equalized_odds(df, A, T):
    parity = dict()


    per_T_result = dict()
    for t in df[T].unique():
        bool_state = True
        for a in df[A].unique():
            pa = cond_proba(df, 'y_pred', 1, [(A, a), (T, t)])
            parity[t, a] = pa
            for b in df[A].unique():
                pb = cond_proba(df, 'y_pred', 1, [(A, b), (T, t)])
                bool_state = bool_state and pa == pb

        per_T_result[t] = bool_state
    
    print(parity)
    print(per_T_result)

    combined = True 
    for k in per_T_result.keys():
        combined = combined and per_T_result[k]

    return combined



# NOTE: A is the demographic group, T the true label
def predictive_value_parity(df, A, T, positive_T_val):
    parity = dict()

    assert len(df[T].unique()) == 2

    negative_T_val = set(list(df[T].unique())) - set([positive_T_val])

    df = df.replace(positive_T_val, 1)
    df = df.replace(negative_T_val, 0)



    per_y_result = dict()
    for y in [1, 0]:
        bool_state = True
        for a in df[A].unique():
            pa = cond_proba(df, T, y, [(A, a), ('y_pred', y)])
            parity[y, a] = pa
            for b in df[A].unique():
                pb = cond_proba(df, T, y, [(A, b), ('y_pred', y)])
                bool_state = bool_state and pa == pb

        per_y_result[y] = bool_state
    
    print(parity)
    print(per_y_result)

    combined = True 
    for k in per_y_result.keys():
        combined = combined and per_y_result[k]

    return combined




if __name__ == '__main__':
    data = []
    for _ in range(45):
        data.append((1, 'peach', 'qualified'))
    for _ in range(45):
        data.append((0, 'peach', 'qualified'))

    for _ in range(2):
        data.append((1, 'peach', 'unqualified'))
    for _ in range(8):
        data.append((0, 'peach', 'unqualified'))

    for _ in range(5):
        data.append((1, 'apple', 'qualified'))
    for _ in range(5):
        data.append((0, 'apple', 'qualified'))

    for _ in range(18):
        data.append((1, 'apple', 'unqualified'))
    for _ in range(72):
        data.append((0, 'apple', 'unqualified'))


    columns = ['y_pred', 'university', 'attribute']
    df = pd.DataFrame(columns=columns, data=data)

    print("demographic parity:")
    print(demographic_parity(df, 'university'))
    print()

    print("equalized odds:")
    print(equalized_odds(df, 'university', 'attribute'))
    print()

    print("predictive value parity:")
    print(predictive_value_parity(df, 'university', 'attribute', 'qualified'))



