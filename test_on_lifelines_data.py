from matplotlib import pyplot as plt
import numpy as np

from stacking import StackingClassifier

from lifelines import datasets
regr = datasets.load_regression_dataset()

subject = np.array([0.11374, 0.40986, 0.064934]) # from dataset

clf = StackingClassifier(model_name="random_forest")
clf.fit(regr, "T", "E")
pred_df= clf.predict_survival_function(subject)
clf.plot_survival_function(pred_df)


