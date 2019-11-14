import os
import pickle


USE_PICKLE = True


BKP_FOLDER = os.path.join("bkp", "run")
os.makedirs(BKP_FOLDER, exist_ok=True)


def use_pickle(func):

    def clean_string(obj):
        return str(obj).replace(' ', '_').replace('{', '') \
            .replace('}', '').replace("'", '').replace(':', '') \
            .replace(',', '').replace('(', '').replace(')', '')

    def dic2string(dic):
        new_dic = {
            k: v for k, v in dic.items()
            if not (hasattr(v, '__len__') and len(v) > 10)
        }

        return clean_string(new_dic)

    def list2string(lst):

        new_lst = [
            v for v in lst
            if not (hasattr(v, '__getitem__') and len(v) > 10)
        ]

        return clean_string(new_lst)

    def inner(*args, **kwargs):

        file_name = f"{func.__name__}_" \
                    f"{list2string(args)}_" \
                    f"{dic2string(kwargs)}" \
                    f".p"

        bkp_file = os.path.join(BKP_FOLDER, file_name)

        # force = 'force' in kwargs and kwargs['force']

        if os.path.exists(bkp_file) and USE_PICKLE:
            data = pickle.load(open(bkp_file, 'rb'))

        else:
            data = func(*args, **kwargs)
            pickle.dump(data, open(bkp_file, 'wb'))

        return data

    return inner
