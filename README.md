# Survival analysis as a classification problem - implementation

This repository contains an implementation of the [following paper](https://arxiv.org/pdf/1909.11171)

## Implementation dependencies

The implementation uses numpy, pandas, and scikit-learn.

## Using the class

Once the `StackingClassifier` class is imported, an instance can be created by invoking the constructor. It has two 
parameters: 

* `model_name`, which is a string, with the default value of "logistic_regression". It is used to determine which base model to use. 
Options are "logistic_regression", "random_forest", "gradient_boosting" and "neural_network". Otherwise, an exception is raised

* `model_args`, a dict which contains keyword arguments to pass to the base model

So for example, instanciating the class using a 2 layer nerual network as a base calssifier can be done by the following call:
 ```python
 clf = StackingClassifier(model_name="neural_network", model_args={"hidden_layer_sizes":(2,)})
 ```

### The `fit` function
The `fit` function is used to fit the model to a dataset. It has the following arguments:
* `event_data`: mandatory argument. A pandas Dataframe. Must contain at least 3 columns. Missing values are not allowed. Must contain only numerical values.
* `time_col`: mandatory argument. The name of the column which contains the time of events in the `event_data` DataFrame. Must contain numeric values greater or equal to zero.
* `event_col`: mandatory argument. The name of the column which contains the events. Can only be 0(meaning the subject loses to follow) or 1(meaning the subject fails to follow).
* `max_censored_set_size`: optional argument. Should be an integer, indicating the maximum size of sensored items in risk sets.

One extra assumption about the data is that a subject cannot rejoin once they lose to follow. The other columns besides `time_col` and `event_col` are considered feature columns.

The fit function creates the stacked data matrix, and fits the model specified in the constructor to it. An example way of calling fit:
```python
e_data = pd.read_csv("regression_data.csv")
clf.fit(e_data, "time", "event")
```

### The `predict` function
The `predict` function is used to predict unseen data. It raises an exception if `fit` wasn't called before on the instance. It has 2 arguments:
* `x_new`:  the feature vector of the unseen data point
* `t`: optional argument. If it is passed, the chance of survival is predicted, at th nearest time after t or which there exists a risk set. In this case, the return value is a tuple 
containing the chance of survival and the standard error. If it is not passed, the whole survival function is returned for every unique data point in the training data. Then the return
value is a tupple of length 3, where the first item is a numpy array containing each unique time stamp in the training data, sorted ascendingly, the second one is the predicted chance of
survival at the given time for the current item, and the last one contains the standard errors. 

An example code to plot the survival funciton:
```python
times, preds, confs = clf.predict([0.11374, 0.40986, 0.064934])
fig, ax = plt.subplots()
ax.plot(times,preds)
ax.fill_between(times, (preds-confs), (preds+confs), color='b', alpha=.1)
plt.show()
```

