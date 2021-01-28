import time
import json
import loader
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier


if __name__ == "__main__":

    # No randomness is introduced in HGBDT when `X_train` and `X_test`
    # are fixed.
    n_jobs = -1
    load_funcs = loader.load_all()
    config = json.load(open("config.json", "r"))
    records = []
    
    for dataset, func in load_funcs.items():

        if dataset in ("sector"):  # run out of memory on sector
            continue

        if dataset not in config:
            msg = "Missing configuration in json file for dataset = {}."
            raise RuntimeError(msg.format(dataset))

        X_train, y_train, X_test, y_test = func()
        n_classes = np.unique(y_train).shape[0]
        objective = ('categorical_crossentropy' if n_classes > 2
                     else 'binary_crossentropy')
        n_trees = config[dataset]
        n_estimators = n_trees // n_classes if objective == 'categorical_crossentropy' else n_trees
        print("Currently processing {}...".format(dataset))
        
        model = HistGradientBoostingClassifier(
            max_iter=n_estimators,
            loss=objective,
            validation_fraction=None
        )

        tic = time.time()
        model.fit(X_train, y_train)
        toc = time.time()
        training_time = toc - tic

        tic = time.time()
        y_pred = model.predict(X_test)
        toc = time.time()
        testing_time = toc - tic
        
        testing_acc = accuracy_score(y_test, y_pred)
        
        records.append((dataset, training_time, testing_time, testing_acc))

    # Write a log file
    with open("all_hgbdt.txt", 'w') as file:
        for dataset, training_time, testing_time, testing_acc in records:
            string = "{}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                dataset, training_time, testing_time, testing_acc)
            file.write(string)
        file.close()
        
