1 - Checking account status: ordinal, 4 categories, Banking behavior proxy for money management, Cramér's V: **0.3570** (medium association) -> **woe9/ enc**

2 - Duration in month: numerical continuous, Loan duration in months, Cohen's d: **-0.5298**(medium) -> standard scaling + 
# Create polynomial
df['duration_squared'] = df['duration'] ** 2
# Create interaction with amount
df['monthly_burden'] = df['credit_amount'] / df['duration']
df['duration_bins'] = pd.qcut(df['duration'], q=5) # optional binning for non-linear models

3 - Credit history: nominal, 5 categories, Weird Finding A34 (critical account/other credits exist) showed GOOD credit performance (contradicts theory!), Cramér's V: **0.2291 (small association)** -> **one hot encoding** (hybrid also applicable woe + critical account flag) merge (no credits/all paid)  one hot due to almost same frequency on two different categories

4 - Purpose: nominal, 10 categories, Cramér's V: 0.1812 (small association) -> merge (education,retraining) and (repairs, other, domestic appliances) -> one hot encoding
df['purpose'] = df['purpose'].replace({'A48': 'A46'})

5 - Credit amount: numerical continuous, Loan amount in DM, Cohen's d: **-0.4232** (medium) -> log transform + standard scaling 
df['monthly_burden'] = df['credit_amount'] / df['duration']
df['amount_tier'] = pd.qcut(df['credit_amount'], q=5) # optional binning for non-linear models

6 - Savings account/bonds: ordinal, 5 categories, Savings = financial cushion for emergencies, Cramér's V: 0.1938 (small association) -> merge (<100> <500>), (>500, >1000) -> woe encoding

7 - Present employment since: ordinal, 5 categories, Employment stability proxy, Cramér's V: 0.1506 (small association) -> ordinal
8 - Installment rate in percentage of disposable income: Numerical but treat as ordinal, 4 categories, non-significant difference + negligible association Cramér's V: 0.0547 (negligible association) -> **drop**


10 - Personal Status Sex: nominal, 4 categories, Cramér's V: 0.0766 (negligible association)-> protected characteristic under ECOA
 -> drop or one hot

 11 - Other debtors / guarantors: nominal, 3 categories, Cramér's V: 0.0640 (negligible association) + non-significant difference, rare labels -> **drop** or one-hot

 12 - present_residence_since:  numerical but treat as ordinal, 4 categories, Cramér's V: 0.0597 (negligible association) + non-significant difference -> **drop** or woe encoding

13 - Property: nominal, 4 categories, Cramér's V: 0.1539 (small association) -> one hot encoding

14 - Age in years: numerical continuous, Cohen's d: 0.1587 (small) -> group into bins + woe 
df['age_groups'] = pd.cut(df['age_years'], 
                          bins=[0, 25, 35, 50, 65, 100],
                          labels=['Young', 'Early_Career', 'Prime', 'Mature', 'Senior'])

15 - Other installment plans: nominal, 3 categories, Store credit = lender of last resort (high correlation with default), Cramér's V: 0.1304 (small association) -> woe

16 - Housing: nominal, 3 categories, Ownership = wealth indicator, cramér's V: 0.1125 (small association) -> one hot encoding frequency

17 - Existing credits count: Discrete numerical treat as ordinal, Multiple credits = higher debt burden, negligible + non-significant + rare labels -> **drop**

18 - Job: ordinal, 4 categories, Employment type proxy, Cramér's V: 0.1135 (small association) -> merge (unskilled, unemployed)  -> woe encoding

19 - people liable for maintenance: discrete numerical treat as ordinal, 3 categories, Higher number = higher financial burden, Cramér's V: 0 + p-value 1 -> **drop**

20 Telephone: **drop**

21 Foreign worker: **drop**
