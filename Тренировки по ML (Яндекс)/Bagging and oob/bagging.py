import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            indices = np.random.randint(0, data_length, data_length)
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))
                   ) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(
            data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            # Your Code Here
            data_bag = np.array([data[i] for i in self.indices_list[bag]])
            target_bag = np.array([target[i] for i in self.indices_list[bag]])
            # store fitted models here
            self.models_list.append(model.fit(data_bag, target_bag))
        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        # Your code here
        self.predictions = np.mean([mod.predict(data)
                                   for mod in self.models_list], axis=0)

        return self.predictions

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here
        for i in range(len(self.data)):
            for mod in range(len(self.models_list)):
                if i not in np.array(self.indices_list[mod]):
                    preds = self.models_list[mod].predict(
                        self.data[i, :].reshape(1, -1))
                    list_of_predictions_lists[i].append(preds.item())

        self.list_of_predictions_lists = np.array(
            list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Your Code Here
        self.indices_mat = np.array(self.indices_list).T
        self.oob_predictions = np.array([None if np.sum(np.sum(self.indices_mat == i, axis=1) == self.num_bags) == 1
                                         else np.nanmean(self.list_of_predictions_lists[i]) for i in range(len(self.data))])

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Your Code Here
        return np.nanmean([(self.oob_predictions[i] - self.target[i])**2 for i in range(len(self.data))
                           if self.oob_predictions[i] is not None])
