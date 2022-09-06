import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


# Contains allowed model names and default parameters
MODELS = {
    "logistic_regression": {
        "model_class": LogisticRegression
    }, 
    "random_forest":{
        "model_class": RandomForestClassifier
    },
    "gradient_boosting":{
        "model_class": GradientBoostingClassifier
    }, 
    "neural_network":{
        "model_class": MLPClassifier
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
    max_sample_size: int, default=None
        Maximum number of censored values to use in risk sets

    Attributes
    ----------
    event_data: pd.DataFrame
        Sorted copy of the DataFrame passed in the constructor
    time_col: string
        Name of the column containing time data
    event_col: string
        Name of the data containing the events
    feature_cols: list of strings
        Contains the name of the remaining columns
    max_sample_size: int
        Maximum number of censored values to use in risk sets
    stratum_data_: pd.DataFrame
        Contains features for each stratum
    """
    def __init__(self, model_name="logistic_regression", model_args={}, max_sample_size=None):
        
        if model_name not in MODELS:
            raise RuntimeError(f"Base model {model_name} not implemented")
        
        model_class = MODELS[model_name]["model_class"]
        
        # extend default params with passed model params
        self.model = model_class(**model_args)
        
        # data fields
        self.event_data = None
        self.time_col = None
        self.event_col = None
        self.feature_cols = None

        self.max_sample_size = max_sample_size
        # store stratum means for inferencing
        self.stratum_data_ = None


    def _set_up_data_fields(self, event_data, time_col, event_col):
        self.time_col = time_col
        self.event_col = event_col
        self.feature_cols = event_data.columns.difference([time_col, event_col])

        # copy and sort event data
        self.event_data = event_data.copy()
        # sort rows by time of the event
        self.event_data = self.event_data.sort_values(by=self.time_col, axis=0)


    def _get_risk_set(self, curr_event_data, event_start_index, event_size):
        """
        Returns risk set. Implements sampling if risk set size is bigger than 
        `max_sample_size` 

        Parameters
        ----------
        curr_event_data: pd.DataFrame
            DataFrame containing the events that happened at the current time
        event_start_index: int
            Index of the first event that happened at the current time in `event_data`
        event_size: int
            Number of events happened at current time
        
        Returns
        ----------
        risk_set: pd.DataFrame
            DataFrame containing the risk set
        """
        censord_set_start_index = event_start_index + event_size
        num_censored = len(self.event_data) - censord_set_start_index

        # sampling is needed if max censored set size is set and it is smaller than
        # the number of censored items in the current risk set
        if self.max_sample_size and (self.max_sample_size < num_censored):
            censored_set= self.event_data.iloc[censord_set_start_index:]
            sampled_censored_set = censored_set.sample(self.max_sample_size, replace=False)
            # concatenate censored set with event data
            return pd.concat([curr_event_data, sampled_censored_set], ignore_index=True)
        else:
            # risk set will be all records after event if subsampling is not needed
            return self.event_data.iloc[event_start_index:]

    
    def _process_stratum_data(self, stratum_data):
        """
        Creates a DataFrame where each record represents a stratum, from a list of dictionaries. 
        The index is the time of the stratum. Also calculates GreenWood coefficients according 
        to Equation 9 in the source paper.

        Parameters
        ----------
        stratum_data_: list of dicts 
            Contains data about each stratum: means of feature columns, number of failures,
            ratio of failures and size of risk set
            
        """
        self.stratum_data_=pd.DataFrame.from_records(stratum_data, index="time")

        # calculate error coeff for stratum
        for t in self.stratum_data_.index:
            events_before_df = self.stratum_data_.loc[:t] # for loc, the last event is included
            y_j = events_before_df.loc[:, "y_j"]
            n_j = events_before_df.loc[:, "n_j"]

            # only sum coeffs where there is no division by zero
            to_sum = y_j/(n_j*(n_j-y_j))
            coeff = np.sum(to_sum[np.isfinite(to_sum)])**(1/2)
            self.stratum_data_.loc[t, "greenwood_coeff"] = coeff


    def _stack_data(self):
        """
        Implements stacking of per stratum data.

        Returns
        ----------
        predictor_mtx: array of shape(n_unique_event_times, n_features)
            Stacked matrix containing centered values, concatenated from each stratum
        response_vectors: array of shape(n_unique_event_times)
            Contains response vectors concatenated from each stratum. Contains 1 where
            the given subject failed at the given time, otherwise 0.
            
        """
        # get each unique time an event happens and number of events
        unique_event_data = np.unique(self.event_data.loc[:, self.time_col], return_index=True, return_counts=True)

        risk_sets, respose_vectors = [], []

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


    def fit(self, event_data, time_col, event_col):
        """
        Fits `model` to passed data.
        
        Parameters
        ----------
        event_data : pd.DataFrame
            Training data.
        time_col: string
            Name of the column containing time data
        event_col: string
            Name of the data containing the events
        
        Returns
        -------
        self : object
            Fitted isntance.
        """
        self._set_up_data_fields(event_data, time_col, event_col)
        
        predictor_mtx, response_vector = self._stack_data()
        self.model.fit(predictor_mtx, response_vector)

        return self
        
    def predict_proba_at(self, x_new, t):
        """
        Predict  conditional probability of surving until the neareast time after t that appeared 
        in the dataset given that x_new has survived until t-1, as per Eq 8. from the paper
        
        Parameters
        ----------
        x_new : array of shape(n_features,)
            Input sample
        t: int, 
            Time at which to predict
        
        Returns
        -------
        p : float
            Chance of survival
        """
        event_times = self.stratum_data_.index.values
        time_idx = np.searchsorted(event_times, t)
        
        # if time is greater than the end of the experiment, return the time at the end of experiment
        if time_idx == len(event_times):
            time_idx = len(event_times)-1

        # get stratum means
        stratum_at = self.stratum_data_.iloc[time_idx]
        prediction = self.model.predict_proba(np.array([x_new - stratum_at.loc[self.feature_cols]]))[0, 1]

        # clip to 0-1 range of probabilities
        chance_of_death = np.clip(prediction + stratum_at.loc["response_vec_mean"], 0, 1) 
        chance_of_survival = 1 - chance_of_death

        return chance_of_survival

    def predict_survival_function(self, x_new):
        """
        Predict survival function using trained `model`.
        
        Parameters
        ----------
        x_new : array of shape(n_features,)
            Input sample
        
        Returns
        -------
        survival_df: pd.Dataframe of shape(n_unique_event_times, 3)
            Contains 3 columns, "time", the times when the event happened,
            "probability", which is the chance of survival until the given time,
            and "std_error" which contains standard error values for the Survival curve 
            based on the Greenwood coefficients
        """
        column_means = self.stratum_data_.loc[:, self.feature_cols]
        target_means = self.stratum_data_.loc[:, "response_vec_mean"]
        greenwood_coeff = self.stratum_data_.loc[:, "greenwood_coeff"]
        # predict using broadcasting, index 1 is chance of death
        predictions = self.model.predict_proba(x_new - column_means)[:, 1]

         # clip to 0-1 range of probabilities
        chances_of_death = np.clip(predictions + target_means, 0, 1) 
        chances_of_survival = 1 - chances_of_death

        # survival is the cumulative product of surviving until time t
        chances_of_survival = np.cumprod(chances_of_survival)
        
        std_error = chances_of_survival * greenwood_coeff
        # return times and chances of survival
        survival_df = pd.DataFrame({
            "time": self.stratum_data_.index.values, 
            "probability": chances_of_survival,
            "std_error": std_error})
        return survival_df

    def plot_survival_function(self, survival_df):
        """
        Plots survival function with standard error
        
        Parameters
        ----------
        survival_df : pd.DataFrame
            Output of predict_survival_function
        """
        times = survival_df.loc[:, "time"]
        preds, errors = survival_df.loc[:, "probability"], survival_df.loc[:, "std_error"]
        
        _fig, ax = plt.subplots()
        ax.set_title("Survival function")
        ax.set_xlabel('Time elapsed')
        ax.set_ylabel('Chance of survival')
        ax.plot(times,preds, label="Survival function")
        ax.fill_between(times, (preds-errors), (preds+errors), 
                        color='b', alpha=.1, label="Std error")
        plt.legend(loc="lower left")
        plt.show()


    