from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Transaction dataset
dataset = [
    ['Milk', 'Bread', 'Butter'],
    ['Milk', 'Diaper', 'Beer', 'Bread'],
    ['Milk', 'Diaper', 'Beer', 'Cola'],
    ['Bread', 'Milk', 'Diaper', 'Beer'],
    ['Bread', 'Milk', 'Diaper', 'Cola']
]

# Convert dataset into one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Step 1: Find frequent itemsets (min support = 0.4)
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# Step 2: Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
