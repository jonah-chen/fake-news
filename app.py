from flask import Flask, request, render_template
from input_test import string_predict_title, string_predict_text

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html', title_score='NA')


@app.route('/text')
def my_form_2():
    return render_template('text.html', text_score='NA')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['title']
    title_score = string_predict_title(text)
    return render_template('index.html', title_score=title_score)


@app.route('/text', methods=['POST'])
def my_form_post_2():
    text = request.form['text']
    text_score = string_predict_text(text)
    return render_template('text.html', text_score=text_score)



if __name__ == '__main__':
    app.run()