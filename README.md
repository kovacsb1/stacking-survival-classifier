# Survival analysis as a classification problem - implementation

This repository contains an implementation of the [following paper](https://arxiv.org/pdf/1909.11171)

## Implementation dependencies

The implementation uses numpy, pandas, scikit-learn and matplotlib.

## Using the class

Once the `StackingClassifier` class is imported, an instance can be created by invoking the constructor. It has 3 optional
parameters: 

* `model_name`, which is a string, with the default value of "logistic_regression". It is used to determine which base model to use. 
Options are "logistic_regression", "random_forest", "gradient_boosting" and "neural_network". Otherwise, an exception is raised

* `model_args`, a dict which contains keyword arguments to pass to the base model

* `max_sample_size`, an int, the maximum number of censored values to use in risk sets

So for example, instantiating the class using a 2 layer neural network as a base classifier can be done by the following call:
 ```python
 clf = StackingClassifier(model_name="neural_network", model_args={"hidden_layer_sizes":(2,)})
 ```

### The `fit` function
The `fit` function is used to fit the model to a dataset. It has the following arguments:
* `event_data`: mandatory argument. A pandas Dataframe. Must contain at least 3 columns. Missing values are not allowed. Must contain only numerical values.
* `time_col`: mandatory argument. The name of the column which contains the time of events in the `event_data` DataFrame. Must contain numeric values greater or equal to zero.
* `event_col`: mandatory argument. The name of the column which contains the events. Can only be 0(meaning the subject loses to follow) or 1(meaning the subject fails to follow).

One extra assumption about the data is that a subject cannot rejoin once they lose to follow. The other columns besides `time_col` and `event_col` are considered feature columns.

The fit function creates the stacked data matrix, and fits the model specified in the constructor to it. An example way of calling fit:
```python
e_data = pd.read_csv("regression_data.csv")
clf.fit(e_data, "time", "event")
```

### The `predict_proba_at` function
The `predict_proba_at` function is used to predict unseen data. It raises an exception if `fit` wasn't called before on the instance. It has 2 arguments:
* `x_new`:  the feature vector of the unseen data point
* `t`: optional argument. The chance of survival is predicted, from the nearest time before t for which exists a risk set  to the nearest time after t, for which exists a risk set. Essentially an implementation of Eq. 8 from the paper

The function returns one value, a probability.


### The `predict_survival_function` function
The `predict_survival_function` function is also used to predict unseen data. It raises an exception if `fit` wasn't called before on the instance. It one argument:
* `x_new`:  the feature vector of the unseen data point

It returns a DataFrame, which contains records for each unique item in the training data for which exists a risk set, and the returned records contain the time, the probability of surviving until that time, and the Greenwood coefficients as per Eq. 9 of the paper.

### The `plot_survival_function` function
Plots survival function with standard error. Has one parameter
* `survival_df`: a pandas DataFrame, containing the output of the `predict_survival_function` function

An example code to plot the survival function:
```python
subject = np.array([0.11374, 0.40986, 0.064934])
pred_df= clf.predict_survival_function(subject)
clf.plot_survival_function(pred_df)
```

