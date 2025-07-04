from sklearn.preprocessing import FunctionTransformer
import numpy as np

def into_1d(x):
    return x.ravel()

flatten = FunctionTransformer(into_1d,validate=False)
