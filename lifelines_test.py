from matplotlib import pyplot as plt
import numpy as np

from stacking import StackingClassifier

from lifelines import datasets, CoxPHFitter
regr = datasets.load_regression_dataset()

subject = np.array([[0.11374, 0.40986, 0.064934]]) # from dataset

cph = CoxPHFitter().fit(regr, 'T', 'E')
times = regr["T"].unique()
survival = cph.predict_survival_function(subject, times=times)
survival = survival.sort_index()

fig, ax = plt.subplots()
ax.plot(survival.index, survival.values, label="Output of Cox PH model")
#plt.show()

clf = StackingClassifier()
clf.fit(regr, "T", "E")
times, preds, confs = clf.predict_proba(subject)


ax.plot(times,preds, label="Output of my implementation")
ax.fill_between(times, (preds-confs), (preds+confs), color='b', alpha=.1, label="Confidence of my prediction")
plt.legend(loc="lower left")
plt.show()

