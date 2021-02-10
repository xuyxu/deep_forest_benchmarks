import time
import json
import loader
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":

    n_jobs = -1
    random_states = [42, 4242, 424242, 42424242, 4242424242]  # 5 trials

    load_funcs = loader.load_regression_all()
    config = json.load(open("config.json", "r"))
    
    for dataset, func in load_funcs.items():

        if dataset not in config:
            msg = "Missing configuration in json file for dataset = {}."
            raise RuntimeError(msg.format(dataset))

        X_train, y_train, X_test, y_test = func()
        n_trees = config[dataset]
        records = []

        for idx, random_state in enumerate(random_states):

            msg = "Currently processing {} with trial {}...".format(
                dataset, idx)
            print(msg)
            
            model = RandomForestRegressor(n_estimators=n_trees,
                                          n_jobs=n_jobs,
                                          random_state=random_state)

            tic = time.time()
            model.fit(X_train, y_train)
            toc = time.time()
            training_time = toc - tic

            tic = time.time()
            y_pred = model.predict(X_test)
            toc = time.time()
            testing_time = toc - tic
            
            testing_mse = mean_squared_error(y_test, y_pred)
            
            records.append((training_time, testing_time, testing_mse))

        # Write a log file
        with open("{}_random_forest_regression.txt".format(dataset), 'w') as file:
            for training_time, testing_time, testing_mse in records:
                string = "{:.5f}\t{:.5f}\t{:.5f}\n".format(training_time,
                                                           testing_time,
                                                           testing_mse)
                file.write(string)
            file.close()
        
