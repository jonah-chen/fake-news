from tensorflow.keras.models import load_model

from preprocessing import to_npy, MAX_LEN, MAX_LEN_TITLE, to_npy_2

def text_predict(file_path, model_path='model_final.h5'):
    f = open(file_path, "r")
    text = f.read()
    f.close()
    # with open(file_path, 'r') as f:
    #     text = f.read().replace('\n', '')
    model = load_model(model_path)

    vec = to_npy(text).reshape(1,MAX_LEN,300)

    a = model(vec, training=False)
    return a

def string_predict_text(text, model_path='model_final.h5'):
    model = load_model(model_path)

    vec = to_npy(text).reshape(1,MAX_LEN,300)

    a = model(vec, training=False)
    b = '%.0f'%(100*float(a[0][1]))+"%"
    return b

def string_predict_title(text, model_path='model_title.h5'):
    model = load_model(model_path)

    vec = to_npy_2(text).reshape(1,MAX_LEN_TITLE,300)

    a = model(vec, training=False)
    b = '%.0f'%(100*float(a[0][1]))+"%"
    return b


def title_predict(file_path, model_path='model_title.h5'):
    f = open(file_path, "r")
    text = f.read()
    f.close()
    # with open(file_path, 'r') as f:
    #     text = f.read().replace('\n', '')
    model = load_model(model_path)

    vec = to_npy_2(text).reshape(1,MAX_LEN_TITLE,300)

    a = model(vec, training=False)
    return a

if __name__ == "__main__":
    # print(text_predict('test_data/test.txt'))
    print(title_predict('test_data/test.txt'))
