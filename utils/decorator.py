import os
import pickle


BKP_FOLDER = os.path.join("bkp", "run")
os.makedirs(BKP_FOLDER, exist_ok=True)


def use_pickle(func):

    """
    Decorator that does the following:
    * If a pickle file corresponding to the call exists,
    it load the data from it instead of calling the function.
    * If no such pickle file exists, it calls 'func',
    creates the file and saves the output in it
    :param func: any function
    :return: output of func(*args, **kwargs)
    """

    def call_func(*args, **kwargs):

        idx_file = os.path.join(BKP_FOLDER, f"{func.__name__}_idx.p")

        info = {k: v for k, v in kwargs.items()}
        info.update({'args': args})

        if os.path.exists(idx_file):

            idx = pickle.load(open(idx_file, 'rb'))
            for i in range(idx):

                info_loaded = pickle.load(
                    open(os.path.join(BKP_FOLDER,
                                      f"{func.__name__}_{i}_info.p"),
                         'rb'))
                print("info", info)
                print("info loaded", info_loaded)
                if info == info_loaded:
                    data = pickle.load(
                        open(os.path.join(BKP_FOLDER,
                                          f"{func.__name__}_{i}_data.p"),
                         'rb'))
                    return data
        else:
            idx = -1

        idx += 1

        data = func(*args, **kwargs)

        data_file = os.path.join(BKP_FOLDER,
                                 f"{func.__name__}_{idx}_data.p")

        info_file = os.path.join(BKP_FOLDER,
                                 f"{func.__name__}_{idx}_info.p")

        pickle.dump(data, open(data_file, 'wb'))
        pickle.dump(info, open(info_file, 'wb'))
        pickle.dump(idx, open(idx_file, 'wb'))

        return data

    return call_func
