import os
import pickle


USE_PICKLE = True


BKP_FOLDER = os.path.join("bkp", "run")
os.makedirs(BKP_FOLDER, exist_ok=True)


def use_pickle(func):

    """
    Decorator that does the following:
    * If a pickle file with the name of '<func name>_<args>_<kwargs>' exists,
    it load the data from it instead of calling the function.
    * If no such pickle file exists, it calls 'func',
    creates the file and saves the output in it
    :param func: any function
    :return: output of func(*args, **kwargs)
    """

    def _clean_string(obj):
        return str(obj).replace(' ', '_').replace('{', '') \
            .replace('}', '').replace("'", '').replace(':', '') \
            .replace(',', '').replace('(', '').replace(')', '')

    def _dic2string(dic):
        new_dic = {
            k: v for k, v in dic.items()
            if not (hasattr(v, '__len__') and len(v) > 10)
        }

        return _clean_string(new_dic)

    def _list2string(lst):

        new_lst = [
            v for v in lst
            if not (hasattr(v, '__getitem__') and len(v) > 10)
        ]

        return _clean_string(new_lst)

    def call_func(*args, **kwargs):

        file_name = f"{func.__name__}_" \
                    f"{_list2string(args)}_" \
                    f"{_dic2string(kwargs)}" \
                    f".p"

        bkp_file = os.path.join(BKP_FOLDER, file_name)

        if os.path.exists(bkp_file) and USE_PICKLE:
            data = pickle.load(open(bkp_file, 'rb'))

        else:
            data = func(*args, **kwargs)
            pickle.dump(data, open(bkp_file, 'wb'))

        return data

    return call_func
