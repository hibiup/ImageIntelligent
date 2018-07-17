from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC


def train_svc(X, y):
    c_vals = [0.0001, 0.01, 0.1, 1, 5, 10, 100, 1000]
    param_grid = [
        {'C': c_vals, 'kernel': ['linear']},
        {'C': c_vals, 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
    ]
    svc = GridSearchCV(SVC(), param_grid, n_jobs=-1)
    svc.fit(X, y)

    return svc
