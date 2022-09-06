import pytest

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from stacking import StackingClassifier

from matplotlib import pyplot as plt

### FIXTURES
@pytest.fixture
def article_data():
    return pd.DataFrame({"x": [1, 0.5, 0], "time":[0,1,2], "event":[1,0,1]})

@pytest.fixture
def toy_data():
    time = np.arange(0, 5, 0.2)
    num_timestamps = len(time)
    
    #linear feature
    x1 = 0.3 * time

    # linear feature plus random noise
    x2 = time + np.random.normal(0, 0.5, num_timestamps)

    # ecponential feature
    x3 = 1.5**time

    events = np.ones(num_timestamps)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "time":time, "event": events})
### END FIXTURES


### TEST CASES
def test_base_model():
    clf = StackingClassifier()
    model = clf.model
    assert isinstance(model, LogisticRegression)

def test_nn_model():
    clf = StackingClassifier(model_name="neural_network", model_args={"hidden_layer_sizes":(2,)})
    model = clf.model
    assert isinstance(model, MLPClassifier)
    assert model.hidden_layer_sizes == (2,)

def test_invalid_model():
    with pytest.raises(RuntimeError):
         clf = StackingClassifier(model_name="xgboost")

def test_stratum_data(article_data):
    clf = StackingClassifier()
    clf.fit(article_data, "time", "event")
    stratum_data = clf.stratum_data_

    assert len(stratum_data) == 2
    
    first_stratum = stratum_data.iloc[0]
    assert first_stratum["x"] == article_data.loc[:, "x"].mean()
    assert first_stratum["y_j"] == 1
    assert first_stratum["n_j"] == 3
    assert first_stratum["response_vec_mean"] == 1/3

    second_stratum = stratum_data.iloc[1]
    assert second_stratum["x"] == article_data.loc[:, "x"].iloc[-1]
    assert second_stratum["y_j"] == 1
    assert second_stratum["n_j"] == 1
    assert second_stratum["response_vec_mean"] == 1

def test_stack_data(article_data):
    clf = StackingClassifier()
    clf._set_up_data_fields(article_data, "time", "event")
    predictor_mtx, response_vector = clf._stack_data()

    assert predictor_mtx.shape == (4,1)
    assert response_vector.shape == (4, )

    input_features = article_data.loc[:, "x"]
    # first stratum should be all values, centered
    assert np.all(predictor_mtx[:3, 0] == (input_features - input_features.mean()))
    # second stratum should be the last feature value
    assert predictor_mtx[3, 0] == input_features.iloc[-1] 
    
    # response vector should look like this
    assert np.all(response_vector == [1,0,0,1])

def test_predict(toy_data):

    clf = StackingClassifier()
    clf.fit(toy_data, "time", "event")

    x_unknown_small = np.array([0,0,0])
    times_small, preds_small, _ = clf.predict_proba(x_unknown_small)

    x_unknown_big = np.array([2,5,15])
    times_big, preds_big, _ = clf.predict_proba(x_unknown_big)

    assert times_small.shape == preds_small.shape
    assert times_small.shape == times_big.shape
    
    # expect survival chances of unknown item with big values to be bigger
    assert np.all(preds_small <= preds_big)

### END TEST CASES