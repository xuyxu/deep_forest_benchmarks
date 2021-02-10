import time
import loader
from sklearn.metrics import accuracy_score

from deepforest import CascadeForestClassifier


if __name__ == "__main__":

    # Hyper-parameters
    n_bins = 255
    bin_subsample = 2e5
    max_layers = 10
    n_estimators = 2
    n_trees = 100
    max_depth = None
    min_samples_leaf = 1
    use_predictor = False
    predictor = "forest"
    n_tolerant_rounds = 2
    partial_mode = True
    delta = 1e-5
    n_jobs = -1
    verbose = 2
    random_states = [42, 4242, 424242, 42424242, 4242424242]  # 5 trials
    load_funcs = loader.load_all()

    for dataset, func in load_funcs.items():

        if dataset in ["mnist", "fashion mnist"]:
            n_trees = 500

        X_train, y_train, X_test, y_test = func()
        records = []

        for idx, random_state in enumerate(random_states):

            msg = "Currently processing {} with trial {}..."
            print(msg.format(dataset, idx))

            model = CascadeForestClassifier(
                n_bins=n_bins,
                bin_subsample=bin_subsample,
                max_layers=max_layers,
                n_estimators=n_estimators,
                n_trees=n_trees,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                use_predictor=use_predictor,
                predictor=predictor,
                n_tolerant_rounds=n_tolerant_rounds,
                partial_mode=partial_mode,
                delta=delta,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
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
            records.append((training_time, testing_time, testing_acc, len(model)))
            model.clean()

        # Writing
        with open("{}_deep_forest_classification.txt".format(dataset), 'w') as file:
            for training_time, testing_time, testing_acc, n_layers in records:
                string = "{:.5f}\t{:.5f}\t{:.5f}\t{}\n".format(
                    training_time, testing_time, testing_acc, n_layers
                )
                file.write(string)
            file.close()
