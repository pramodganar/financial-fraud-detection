import joblib


def save_object(file_path, obj):

    joblib.dump(obj, file_path)


def load_object(file_path):

    return joblib.load(file_path)