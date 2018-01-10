import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from pipe import pipe_lr
from common import X_train, X_test, y_train, y_test


kfold = StratifiedKFold(y=y_train, n_folds=10)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))


#----------------------------#
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))