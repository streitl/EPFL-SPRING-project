import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.loader import load_dataset
from src.models import SRR
from src.preprocessing import one_hot_encode, processing_pipeline
from src.feature_selection import forward_stepwise_regression
from src.vulnerabilities import find_adversarial_examples

dataset = 'german_credit'
print(f"-> {dataset} dataset")
# Load the data
X, y = load_dataset(name=dataset)

# Apply the processing pipeline
X_train_bin, X_test_bin, y_train, y_test = processing_pipeline(X, y)


# One-hot encode the categorical variables
X_train_one_hot = one_hot_encode(X_train_bin)
X_test_one_hot = one_hot_encode(X_test_bin)

# Construct and train Select-Regress-Round model
model = SRR(k=3, M=5)
model.fit(X_train_one_hot, y_train, verbose=False)

# Show statistics of the model
train_acc = accuracy_score(y_train, model.predict(X_train_one_hot)) * 100
test_acc = accuracy_score(y_test, model.predict(X_test_one_hot)) * 100
baseline = max(1-y.mean(), y.mean()) * 100
print(f"Training accuracy of {train_acc:.1f} % and test accuracy of {test_acc:.1f} % (baseline {baseline:.1f} %)\n")
print(model.df)

interval_mapping = {
    pd.Interval(left=float("-inf"), right=11.5): 0,
    pd.Interval(left=11.5, right=23): 1,
    pd.Interval(left=23, right=float("inf")): 2
}

account_mapping = {
    'A11': '... < 0 DM',
    'A12': '0 <= ... < 200 DM',
    'A13': '... >= 200 DM / salary assignments for at least 1 year',
    'A14': 'no checking account'
}

history_mapping = {
    'A30': 'no credits taken/ all credits paid back duly',
    'A31': 'all credits at this bank paid back duly',
    'A32': 'existing credits paid back duly till now',
    'A33': 'delay in paying off in the past',
    'A34': 'critical account/ other credits existing (not at this bank)'
}


adversaries = find_adversarial_examples(model, X_train_bin, y_train, ['Duration_in_months'], unit_changes=True)

can_change = 'Duration_in_months'
others = list(model.df.index.levels[0])
others.remove(can_change)
others.remove('bias')

df = adversaries['adversarial'].copy()
df = df[df.label == 1].drop(columns='label')
df['change'] = df[can_change].replace(interval_mapping)\
                - adversaries['original'][adversaries['original']['label'] == 0][can_change].replace(interval_mapping).astype(int)
df = df.groupby(others + ['change']).size().rename('amount').reset_index()

# Add extra zeroed-entries with changes 0, 1 and 2 so that these values appear on the graph
df.loc[12] = ('A30', 'A11', 0, 0)
df.loc[13] = ('A30', 'A11', 1, 0)
df.loc[14] = ('A30', 'A11', 2, 0)


ax = sns.catplot(x='change', y='amount',
                 col=others[0],
                 row=others[1],
                 data=df,
                 margin_titles=True,
                 height=3,
                 kind='bar',
                 palette='Spectral')

acc_desc = "Status_of_checking_account"
for k, v in account_mapping.items():
    acc_desc += f'\n{k}: {v}'
ax.fig.text(s=acc_desc,  x=0, y=0.8);

cred_desc = "Credit_history:"
for k, v in history_mapping.items():
    cred_desc += f'\n{k}: {v}'
ax.fig.text(s=cred_desc, x=0.6, y=0.8)

ax.fig.set_figheight(8)
plt.subplots_adjust(top=0.75)
ax.fig.suptitle(f"""[german_credit] Count of adversarial examples changing decision from 'bad' to 'good',
            as well as relative interval change, when only '{can_change}' is changed (per feature).""", size=16);
plt.show()



train = X_train_bin.copy()
train['label'] = y_train

ax = sns.catplot(x='label', y='amount',
                 col=others[0],
                 row=others[1],
                 data=train.groupby(['label'] + others).size().rename('amount').reset_index(),
                 margin_titles=True,
                 height=3,
                 kind='bar')

acc_desc = "Status_of_checking_account"
for k, v in account_mapping.items():
    acc_desc += f'\n{k}: {v}'
ax.fig.text(s=acc_desc,  x=0, y=0.89);

cred_desc = "Credit_history:"
for k, v in history_mapping.items():
    cred_desc += f'\n{k}: {v}'
ax.fig.text(s=cred_desc, x=0.6, y=0.89)


ax.fig.set_figheight(14)
plt.subplots_adjust(top=0.85)
ax.fig.suptitle(f"[german_credit] Number of positive and negative training datapoints for each (Credit_history, Status_of_checking_account) pair.", size=16);
plt.show()


ax = sns.catplot(x='change', y='amount',
            data=df.groupby('change')[['amount']].sum().reset_index(),
            margin_titles=True, height=3, kind='bar',
            palette='Spectral')

plt.subplots_adjust(top=0.8)
ax.fig.suptitle(f"[german_credit] Count of adversarial examples changing decision from 'bad' to 'good',\nas well as relative interval change, when only '{can_change}' is changed (all features).", size=12);
plt.show()
