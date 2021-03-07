from tensorflow.keras.models import load_model

from preprocessing import to_npy, MAX_LEN, MAX_LEN_TITLE, to_npy_2

def text_predict(file_path, model_path):
    f = open(file_path, "r")
    text = f.read()
    f.close()
    # with open(file_path, 'r') as f:
    #     text = f.read().replace('\n', '')
    model = load_model(model_path)

    vec = to_npy(text).reshape(1,MAX_LEN,300)

    return model(vec, training=False)

def title_predict(file_path, model_path):
    f = open(file_path, "r")
    text = f.read()
    f.close()
    # with open(file_path, 'r') as f:
    #     text = f.read().replace('\n', '')
    model = load_model(model_path)

    vec = to_npy_2(text).reshape(1,MAX_LEN_TITLE,300)

    return model(vec, training=False)

if __name__ == "__main__":
    print(title_predict('test_data/test.txt', 'model_full.h5'))