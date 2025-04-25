from flask import Flask, request, render_template
from tokenizer_file import process_prompt, detect_intent

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        intent = detect_intent(prompt)
        if intent == "on":
            image = "light_on.jpg"
        elif intent == "off":
            image = "light_off.jpg"
        else:
            image = "default.jpg"
        return render_template('index.html', image=image)
    return render_template('index.html', image="default.jpg")

if __name__ == '__main__':
    app.run(debug=True)
