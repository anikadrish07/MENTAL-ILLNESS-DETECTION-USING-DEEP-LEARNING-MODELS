# app.py
from flask import Flask, render_template, request, session, redirect, url_for
from model import load_data, split_data, vectorize_text, encode_labels, train_model, print_prediction

app = Flask(__name__)
app.secret_key = "supersekrit"

# Load data
df = load_data('combined_data.csv')

# Split data
train, val, test = split_data(df)

# Vectorize text
X_train, X_val, X_test, vectorizer = vectorize_text(train['text'], val['text'], test['text'])

# Encode labels
y_train, y_val, y_test, label_encoder = encode_labels(train['target'], val['target'], test['target'])

# Train model
clf, _, _, _ = train_model(X_train, y_train, X_val, y_val, X_test, y_test, random_state=0)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/learn-about-mental-health')
def learn_about_mental_health():
    return render_template('types.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')


# Route for the first question
@app.route('/question1', methods=['GET', 'POST'])
def question1():
    if request.method == 'POST':
        session['question1'] = request.form['answer']
        return redirect('/question2')  # Redirect to the next question
    return render_template('question-page-1.html')

# Route for the second question
@app.route('/question2', methods=['GET', 'POST'])
def question2():
    if request.method == 'POST':
        session['question2'] = request.form['answer']
        return redirect('/question3')  # Redirect to the next question
    return render_template('question-page-2.html')

# Route for the third question
@app.route('/question3', methods=['GET', 'POST'])
def question3():
    if request.method == 'POST':
        session['question3'] = request.form['answer']
        return redirect('/question4')  # Redirect to the next question
    return render_template('question-page-3.html')

# Route for the fourth question
@app.route('/question4', methods=['GET', 'POST'])
def question4():
    if request.method == 'POST':
        session['question4'] = request.form['answer']
        return redirect('/question5')  # Redirect to the next question
    return render_template('question-page-4.html')

@app.route('/question5', methods=['GET', 'POST'])
def question5():
    if request.method == 'POST':
        # Collect all answers from previous questions
        answers = {f'question{i}': session.get(f'question{i}', '') for i in range(1, 6)}
        # Concatenate all answers into a single text
        concatenated_answers = ' '.join(answers.values())
        session['question5'] = request.form['answer']  # Store the answer for question5
        # Make prediction using the model
        prediction = print_prediction(concatenated_answers, clf, vectorizer, label_encoder)
        return render_template('landing-page.html', prediction=prediction)
    return render_template('question-page-5.html')


if __name__ == '__main__':
    app.run(debug=True)





























# from flask import Flask, render_template, request, session, redirect, url_for
# from model import split_data, vectorize_text, encode_labels, train_model, print_prediction, load_data

# app = Flask(__name__)
# app.secret_key = "supersekrit"  # Change this to your secret key


#     # Load data
# downsampled_df = (load_data('combined_data.csv'))

#     # Split data
# train, val, test = split_data(downsampled_df)

#     # Vectorize text
# X_train, X_val, X_test, vectorizer = vectorize_text(train['text'], val['text'], test['text'])

#     # Encode labels
# y_train, y_val, y_test, label_encoder = encode_labels(train['target'], val['target'], test['target'])

#     # Train model
# clf = train_model(X_train, y_train, X_val, y_val, X_test, y_test, random_state=0)



# @app.route('/')
# def index():
#     return render_template('index.html')


# # Route for the first question
# @app.route('/question1', methods=['GET', 'POST'])
# def question1():
#     if request.method == 'POST':
#         session['question1'] = request.form['answer']
#         return redirect('/question2')  # Redirect to the next question
#     return render_template('question1.html')

# # Route for the second question
# @app.route('/question2', methods=['GET', 'POST'])
# def question2():
#     if request.method == 'POST':
#         session['question2'] = request.form['answer']
#         return redirect('/question3')  # Redirect to the next question
#     return render_template('question2.html')

# # Route for the third question
# @app.route('/question3', methods=['GET', 'POST'])
# def question3():
#     if request.method == 'POST':
#         session['question3'] = request.form['answer']
#         return redirect('/question4')  # Redirect to the next question
#     return render_template('question3.html')

# # Route for the fourth question
# @app.route('/question4', methods=['GET', 'POST'])
# def question4():
#     if request.method == 'POST':
#         session['question4'] = request.form['answer']
#         return redirect('/question5')  # Redirect to the next question
#     return render_template('question4.html')


# # Route for the fifth question
# @app.route('/question5', methods=['GET', 'POST'])
# def question5():
#     if request.method == 'POST':
#         session['question5'] = request.form['answer']
#         # Collect all answers
#         answers = {
#             'question1': session.get('question1', ''),
#             'question2': session.get('question2', ''),
#             'question3': session.get('question3', ''),
#             'question4': session.get('question4', ''),
#             'question5': session.get('question5', '')
#         }

#         # Check if any answer is missing
#         if '' in answers.values():
#             return render_template('index.html')  # Render a page indicating incomplete answers
        
#         # Concatenate all answers into a single text
#         concatenated_answers = ' '.join(answers.values())
        
#             # Make prediction using the model
#         prediction = print_prediction(concatenated_answers, clf, vectorizer, label_encoder)
#             # Pass the prediction to the template
#         return render_template(url_for('landing'), prediction=prediction)
        
#     return render_template('question5.html')


# if __name__ == '__main__':
#     app.run(debug=True)
