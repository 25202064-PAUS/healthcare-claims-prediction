"""Portable scorer for the packaged binary Gradient Boosting model.
No scikit-learn/joblib pickle loading is required at runtime.
"""
import json
import math
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'models' / 'fraud_model_portable.json'
with MODEL_PATH.open('r', encoding='utf-8') as f:
    MODEL = json.load(f)

FEATURES = MODEL['features']

def _transformed_row(row):
    values = []
    for col, median in zip(MODEL['numeric_columns'], MODEL['numeric_medians']):
        value = row.get(col, median)
        if value is None or value == '':
            value = median
        values.append(float(value))
    for col, mode, categories in zip(MODEL['categorical_columns'], MODEL['categorical_modes'], MODEL['categories']):
        value = row.get(col, mode)
        if value is None or value == '':
            value = mode
        for category in categories:
            values.append(1.0 if value == category else 0.0)
    return values

def _tree_value(tree, x):
    node = 0
    while tree['left'][node] != -1:
        feature = tree['feature'][node]
        if x[feature] <= tree['threshold'][node]:
            node = tree['left'][node]
        else:
            node = tree['right'][node]
    return tree['value'][node]

def predict_fraud_probability(row):
    x = _transformed_row(row)
    raw = MODEL['init_raw']
    for tree in MODEL['trees']:
        raw += MODEL['learning_rate'] * _tree_value(tree, x)
    if raw >= 0:
        z = math.exp(-raw)
        return 1.0 / (1.0 + z)
    z = math.exp(raw)
    return z / (1.0 + z)
