from flask import Flask, render_template, request, redirect, url_for, session
from flask_dance.contrib.google import make_google_blueprint, google

from model import preprocess_data, split_data, vectorize_text, encode_labels, train_model, print_prediction, load_data

app = Flask(__name__)
app.secret_key = "supersekrit"  # Change this to your secret key

# Configure Google OAuth
google_bp = make_google_blueprint(
    client_id="your-google-client-id",
    client_secret="your-google-client-secret",
    redirect_to="google_login"
)
app.register_blueprint(google_bp, url_prefix="/login")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = print_prediction(text, vectorizer, clf, label_encoder)
        return render_template('index.html', text=text, prediction=prediction)


@app.route('/login')
def login():
    if not google.authorized:
        return redirect(url_for("google.login"))
    resp = google.get("/oauth2/v1/userinfo")
    assert resp.ok, resp.text
    email = resp.json()["email"]
    session["email"] = email  # You can store user information in the session
    return redirect(url_for("index"))

if __name__ == '__main__':
    # Load data
    df = preprocess_data(load_data('Mental-health-related-subreddits.csv'))

    # Split data
    train, val, test = split_data(df)
    
    # Vectorize text
    X_train, X_val, X_test, vectorizer = vectorize_text(train['text'], val['text'], test['text'])
    
    # Encode labels
    y_train, y_val, y_test, label_encoder = encode_labels(train['target'], val['target'], test['target'])
    
    # Train model
    clf = train_model(X_train, y_train)

    app.run(debug=True)
