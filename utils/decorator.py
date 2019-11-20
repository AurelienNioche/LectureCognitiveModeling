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

        info_file = os.path.join(BKP_FOLDER, f"{func.__name__}_info.p")

        key = kwargs.copy().update({'args': args})

        if os.path.exists(info_file):

            info = pickle.load(open(info_file, 'rb'))
            if key in info.keys() and os.path.exists(info[key]):
                data = pickle.load(open(info[key], 'rb'))
                return data
        else:
            info = {'idx': -1}

        data = func(*args, **kwargs)

        bkp_file = os.path.join(BKP_FOLDER,
                                f"{func.__name__}{info['idx'] + 1}.p")
        pickle.dump(data, open(bkp_file, 'wb'))
        pickle.dump(info, open(info_file, 'wb'))

        return data

    return call_func
