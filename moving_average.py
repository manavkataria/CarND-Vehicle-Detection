import pandas as pd
import numpy as np


class MovingAverage(object):

    def __init__(self, columns, size=5):
        self.df = pd.DataFrame(columns=columns)
        self.size = size     # Sliding Memory Window Size
        self.counter = 0     # Current Index Pointer/Conuter

    def moving_average(self, data):
        """
        Returns a Moving Average _after_ adding the current data to the series.
        data is a dict with column keys to be added to memory
        """
        idx = self.counter % self.size
        self.df.loc[idx] = data

        columns = self.df.columns.values
        moving_average = pd.DataFrame(columns=columns)

        try:
            for col in columns:
                moving_average[col] = self.df[col].values.mean(keepdims=True)
        except:
            import ipdb; ipdb.set_trace()
            moving_average = None

        self.counter += 1
        return moving_average


def test_new_ma():
    data = {
        'scalar': 1,
        '1d': np.array([1, 2, 3, 4]),
        '2d': np.array([[1, 1],[2, 2]])
    }

    ma = MovingAverage(columns=data.keys())
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


def test_array_ma():
    data = {
        'scalar': [1,],
        '1d': [np.array([1, 2, 3, 4]),]
    }

    df = pd.DataFrame(data)

    print(df)

    # Genereate New Values
    for key, values in data.items():
        data[key].append(values[-1]*2)

    df = pd.DataFrame(data)

    # Genereate New Values
    for key, values in data.items():
        data[key].append(values[-1]*2)

    # Genereate New Values
    for key, values in data.items():
        data[key].append(values[-1]*2)

    newdf = pd.DataFrame(data)
    import ipdb; ipdb.set_trace()

    rm = newdf.rolling(window=2,center=False).mean()

    print(rm)


def main():
    test_new_ma()


if __name__ == '__main__':
    main()
