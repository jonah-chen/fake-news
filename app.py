from flask import Flask, request, render_template
from input_test import string_predict

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html', test='hello world')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = string_predict(text, 'model-title.h5')
    return render_template('index.html', test=processed_text)

if __name__ == '__main__':
    app.run()