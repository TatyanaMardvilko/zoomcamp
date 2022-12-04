#!/usr/bin/env python
# coding: utf-8


import pickle


def load(file_name):
    with open(file_name, 'rb') as f_in:
        return pickle.load(f_in)


model = load('model1.bin')
dv = load('dv.bin')

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

X = dv.transform(client)

y_pred = model.predict_proba(X)[0, 1]

print(y_pred)  # Q3 0.162
