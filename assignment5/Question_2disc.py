import pandas, numpy, random, sys, sklearn.ensemble as ensemble, sklearn.metrics as metrics, statsmodels.api as stats, seaborn as sns, matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import (auc, roc_curve)

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.options.display.float_format = '{:,.10}'.format

rand_state = 20221225

train_data = pandas.read_csv('./WineQuality_Train.csv', delimiter=',')
train_data = train_data.dropna()
train_data_size = train_data.groupby('quality_grp').size()

test_data = pandas.read_csv('./WineQuality_Test.csv', delimiter=',')
test_data = train_data.dropna()
test_data_size = train_data.groupby('quality_grp').size()

X_name = ['alcohol', 'free_sulfur_dioxide', 'sulphates', 'citric_acid', 'residual_sugar']

# Build a logistic regression
y_train = train_data['quality_grp'].astype('category')
y_category = y_train.cat.categories
X_train = train_data[X_name]
X_train = stats.add_constant(X_train, prepend=True)

y_test = test_data['quality_grp'].astype('category')
X_test = test_data[X_name]
X_test = stats.add_constant(X_test, prepend=True)

logit = stats.MNLogit(y_train, X_train)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(maxiter = 100)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))

y_predProb = thisFit.predict(X_train)
y_predict = pandas.to_numeric(y_predProb.idxmax(axis=1))

##calculate the AUC metric of the Training data
y_predProb = thisFit.predict(X_train)[1]
fpr, tpr, thresholds = roc_curve(y_train, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('Train AUC =', AUC)

##calculate the AUC metric of the Testing data
y_predProb = thisFit.predict(X_test)[1]
fpr, tpr, thresholds = roc_curve(y_train, y_predProb, pos_label=1)
AUC = auc(fpr, tpr)
print('Test AUC =', AUC)

y_predictClass = y_category[y_predict]
y_confusion = metrics.confusion_matrix(y_train, y_predictClass)
print("Confusion Matrix (Row is Data, Column is Predicted) = \n")
print(y_confusion)
y_accuracy = metrics.accuracy_score(y_train, y_predictClass)
print("Accuracy Score = ", y_accuracy)

AUCs = []
n_iterations = 100000
for i in range(n_iterations):
    X_bs, y_bs = resample(X_train, y_train, replace=True, random_state=rand_state)
    # make predictions
    y_predProb = thisFit.predict(X_bs)[1]
    fpr, tpr, thresholds = roc_curve(y_train, y_predProb, pos_label=1)
    AUCs.append(auc(fpr, tpr))
    
# plot distribution of AUC
sns.histplot(AUCs, binwidth=0.001)
plt.title("AUC across 100,000 bootstrap samples")
plt.xlabel("AUC")
plt.show()

##calculate percentiles
percentiles = numpy.percentile(AUCs, [2.5, 50, 97.5])
print(percentiles)