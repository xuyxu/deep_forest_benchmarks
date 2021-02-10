import time
import json
import loader
import numpy as np
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor


if __name__ == "__main__":

    # No randomness is introduced in LightGBM when `X_train` and `X_test`
    # are fixed.
    n_jobs = -1
    load_funcs = loader.load_regression_all()
    config = json.load(open("config.json", "r"))
    records = []
    
    for dataset, func in load_funcs.items():

        if dataset not in config:
            msg = "Missing configuration in json file for dataset = {}."
            raise RuntimeError(msg.format(dataset))

        X_train, y_train, X_test, y_test = func()
        n_classes = np.unique(y_train).shape[0]
        n_estimators = config[dataset]
        print("Currently processing {}...".format(dataset))
        
        model = LGBMRegressor(boosting_type='gbdt',
                              n_estimators=n_estimators,
                              n_jobs=n_jobs)

        tic = time.time()
        model.fit(X_train, y_train)
        toc = time.time()
        training_time = toc - tic

        tic = time.time()
        y_pred = model.predict(X_test)
        toc = time.time()
        testing_time = toc - tic
        
        testing_mse = mean_squared_error(y_test, y_pred)
        
        records.append((dataset, training_time, testing_time, testing_mse))

    # Write a log file
    with open("all_lightgbm_regression.txt", 'w') as file:
        for dataset, training_time, testing_time, testing_mse in records:
            string = "{}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                dataset, training_time, testing_time, testing_mse)
            file.write(string)
        file.close()
        
