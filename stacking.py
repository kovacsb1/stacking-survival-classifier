import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


MODELS = {
    "logistic_regression": {
        "model_class": LogisticRegression, 
        "default_params": {"fit_intercept": False}
    }, 
    "random_forest":{
        "model_class": RandomForestClassifier, 
        "default_params": {},
    },
    "gradient_boosting":{
        "model_class": GradientBoostingClassifier,
        "default_params": {}
    }, 
    "neural_network":{
        "model_class": MLPClassifier,
        "default_params": {}
    }
}
        

class StackingClassifier:
    """
    Stacking classifier for survival analysis, implemented from https://arxiv.org/pdf/1909.11171.

    Parameters
    ----------
    model_name: string, default="logistic_regression"
        which base model to use. Options are logistic_regression, random_forest, gradient_boosting
        and neural_network. Otherwise, raises exception
    model_args: dict
        contains keyword arguments to pass to the base model

    Attributes
    ----------


    """
    def __init__(self, model_name="logistic_regression", model_args={}):
        
        if model_name in MODELS:
            model_class = MODELS[model_name]["model_class"]
            model_params = MODELS[model_name]["default_params"]
            
            # extend default params with passed model params
            concatenated_params = dict(model_params, **model_args)
            self.model = model_class(**concatenated_params)

        # data fields
        self.event_data = None
        self.time_col = None
        self.event_col = None
        self.feature_cols = None

        self.max_censored_set_size = None

        # store stratum means for inferencing
        self.stratum_data = None


    def _set_up_data_fields(self, event_data, time_col, event_col, max_censored_set_size):
        self.time_col = time_col
        self.event_col = event_col
        self.feature_cols = event_data.columns.difference([time_col, event_col])

        # copy and sort event data
        self.event_data = event_data.copy()
        # sort rows by time of the event
        self.event_data = self.event_data.sort_values(by=self.time_col, axis=0)

        self.max_censored_set_size = max_censored_set_size


    def _get_risk_set(self, curr_event_data, event_start_index, event_size):
        censord_set_start_index = event_start_index + event_size
        num_censored = len(self.event_data) - censord_set_start_index

        # sampling is needed if max censored set size is set and it is smaller than
        # the number of censored items in the current risk set
        if self.max_censored_set_size and (self.max_censored_set_size < num_censored):
            censored_set= self.event_data.iloc[censord_set_start_index:]
            sampled_censored_set = censored_set.sample(self.max_censored_set_size, replace=False)
            # concatenate censored set with event data
            return pd.concat([curr_event_data, sampled_censored_set], ignore_index=True)
        else:
            # risk set will be all records after event if subsampling is not needed
            return self.event_data.iloc[event_start_index:]

    
    def _process_stratum_data(self, stratum_data):
        self.stratum_data=pd.DataFrame.from_records(stratum_data, index="time")

        # calculate error coeff for stratum
        for t in self.stratum_data.index:
            events_before_df = self.stratum_data.loc[:t] # for loc, the last event is included
            y_j = events_before_df.loc[:, "y_j"]
            n_j = events_before_df.loc[:, "n_j"]

            # only sum coeffs where there is no division by zero
            to_sum = y_j/(n_j*(n_j-y_j))
            coeff = np.sum(to_sum[np.isfinite(to_sum)])**(1/2)
            self.stratum_data.loc[t, "greenwood_coeff"] = coeff


    def _stack_data(self):
        # get each unique time an event happens and number of events
        unique_event_data = np.unique(self.event_data.loc[:, self.time_col], return_index=True, return_counts=True)

        risk_sets, respose_vectors = [], []
        #column_means, target_means = {}, {}
        stratum_data = []
        # iterate through each unique event time
        for e_time, e_idx, e_count in zip(*unique_event_data):
            curr_event_data = self.event_data.iloc[e_idx:e_idx+e_count]
            failed_mask = curr_event_data.loc[:, self.event_col] == 1

            # has any failed during this time
            if failed_mask.any():
                risk_set = self._get_risk_set(curr_event_data, e_idx, e_count)

                # get feature columns and center them
                risk_features = risk_set.loc[:, self.feature_cols]
                mean_risk = risk_features.mean(axis=0)
                centered_risk_features = risk_features - mean_risk
                risk_sets.append(centered_risk_features.values)

                # response vector will be copy of events at current time with rest of the risk set to zero
                respose_vector = risk_set.loc[:, self.event_col].copy()
                respose_vector.iloc[e_count: ] = 0
                respose_vectors.append(respose_vector)

                # store properties of stratum
                feature_mean_dict = mean_risk.to_dict()
                feature_mean_dict.update({
                    "time": e_time,
                    "y_j": respose_vector.sum(),
                    "n_j": len(respose_vector),
                    "response_vec_mean": respose_vector.mean()
                })
                stratum_data.append(feature_mean_dict)

        self._process_stratum_data(stratum_data)

        return np.concatenate(risk_sets), np.concatenate(respose_vectors)


    def fit(self, event_data, time_col, event_col, max_censored_set_size=None):
        self._set_up_data_fields(event_data, time_col, event_col, max_censored_set_size)
        
        predictor_mtx, response_vectors = self._stack_data()
        self.model.fit(predictor_mtx, response_vectors)
        

    def predict_one(self, x_new):
        column_means = self.stratum_data.loc[:, self.feature_cols]
        target_means = self.stratum_data.loc[:, "response_vec_mean"]
        greenwood_coeff = self.stratum_data.loc[:, "greenwood_coeff"]
        # predict using broadcasting, index 1 is chance of death
        predictions = self.model.predict_proba(x_new - column_means)[:, 1]

         # clip to 0-1 range of probabilities
        chances_of_death = np.clip(predictions+target_means, 0, 1)
        chance_of_survival = 1 - chances_of_death
        std_error = chance_of_survival * greenwood_coeff
        # return times and chances of survival
        return self.stratum_data.index.values, chance_of_survival, std_error


    def predict(self, x_new, t):
        event_times = self.stratum_data.index.values
        time_idx = np.searchsorted(event_times, t)
        
        # if time is greater than the end of the experiment, return the time at the end of experiment
        if time_idx == len(event_times):
            time_idx = len(event_times)-1

        # get stratum means
        stratum_at = self.stratum_data.iloc[time_idx]

        prediction = self.model.predict_proba(np.array([x_new - stratum_at.loc[self.feature_cols]]))[0, 1]

        # clip to 0-1 range of probabilities
        chance_of_death = np.clip(prediction + stratum_at.loc["response_vec_mean"], 0, 1) 
        chance_of_survival = 1 - chance_of_death

        std_error = chance_of_survival * stratum_at["greenwood_coeff"]
        return chance_of_survival, chance_of_survival * std_error