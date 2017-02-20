#!/usr/bin/env ipython
import pandas as pd
import numpy as np
import copy

from utils import debug


class MovingAverage(object):

    def __init__(self, columns, size=5):
        self.df = pd.DataFrame(columns=columns)
        self.size = size     # Sliding Memory Window Size
        self.counter = 0     # Current Index Pointer/Conuter
        self.incompatible_keys = []     # Bookkeeping for back and forth type conversions

    def _to_numpy(self, data):
        d = copy.deepcopy(data)
        self.incompatible_keys = []
        for key, value in d.items():
            if isinstance(value, list):
                d[key] = np.array(value)
                self.incompatible_keys.append(key)
        return d

    def _to_list(self, data):
        for key, value in data.items():
            if key in self.incompatible_keys and isinstance(value, np.ndarray):
                data[key] = value.tolist()
        self.incompatible_keys = []
        return data

    def moving_average(self, data):
        """
        Returns a Moving Average _after_ adding the current data to the series.
        data is a dict with column keys to be added to memory
        """

        data = self._to_numpy(data)

        idx = self.counter % self.size
        self.df.loc[idx] = pd.Series(data)

        columns = self.df.columns.values
        moving_average = pd.DataFrame(columns=columns)

        try:
            for col in columns:
                moving_average[col] = self.df[col].values.mean(keepdims=True)
        except Exception:
            debug('Error! Unable to compute moving average: ' + Exception)
            import ipdb; ipdb.set_trace()
            moving_average = None
        finally:
            self.counter += 1

        ma_dict = moving_average.to_dict(orient='records')[0]
        return self._to_list(ma_dict)


def test_ma_size_1():
    data = {
        'scalar': 1,
        '1d': np.array([1, 2, 3, 4]),
        '2d': np.array([[1, 1],[2, 2]])
    }

    ma = MovingAverage(columns=data.keys(), size=1)
    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

def test_ma_size_5():
    data = {
        'scalar': 1,
        '1d': np.array([1, 2, 3, 4]),
        '2d': np.array([[1, 1],[2, 2]])
    }

    ma = MovingAverage(columns=data.keys(), size=5)
    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*2

    print(ma.moving_average(data))

def test_ma_list():
    data = {
        'scalar': 1,
        '1d': [1, 2, 3, 4],
        '2d': [[1, 1],[2, 2]]
    }

    ma = MovingAverage(columns=data.keys(), size=1)
    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*1

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*1

    print(ma.moving_average(data))

    # Genereate New Values
    for key, values in data.items():
        data[key] = values*1

    print(ma.moving_average(data))


def main():
    test_ma_list()


if __name__ == '__main__':
    main()
