from flask import Flask, render_template, request, jsonify
import nltk
import string
import re
import random
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required nltk data (run once)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

app = Flask(__name__,template_folder='template')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Sample training data (same as before)
training_data = [
    ("Hello", "greeting"),
    ("Hi there", "greeting"),
    ("Good morning", "greeting"),
    ("Goodbye", "farewell"),
    ("See you later", "farewell"),
    ("Thanks a lot", "thanks"),
    ("Thank you", "thanks"),
    ("Can you help me?", "help"),
    ("I want information about your product", "product_info"),
    ("Tell me about your services", "product_info"),
    ("What are your business hours?", "hours"),
    ("When are you open?", "hours"),
]

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(t) for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(filtered)

# Prepare data
texts = [preprocess_text(sentence) for sentence, intent in training_data]
labels = [intent for sentence, intent in training_data]

# Vectorization and model training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
model = LogisticRegression()
model.fit(X, labels)

def extract_entities(text):
    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\b\d{10}\b', text)
    return {"email": email, "phone": phone}

responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?"],
    "farewell": ["Goodbye! Have a great day.", "See you later!"],
    "thanks": ["You're welcome!", "Happy to help!"],
    "help": ["Sure, I can help! Please ask your question."],
    "product_info": ["We offer a range of products tailored to your needs.", "Please specify which product you'd like to know more about."],
    "hours": ["We are open from 9 AM to 6 PM, Monday to Friday."],
}

def chatbot_response(user_text):
    processed = preprocess_text(user_text)
    vector = vectorizer.transform([processed])
    intent = model.predict(vector)[0]
    entities = extract_entities(user_text)
    response = random.choice(responses.get(intent, ["Sorry, I didn't understand that. Can you please rephrase?"]))
    if entities["email"]:
        response += f" I noticed you mentioned an email: {entities['email'][0]}"
    if entities["phone"]:
        response += f" I noticed you mentioned a phone number: {entities['phone']}"
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_msg = request.form["message"]
    bot_reply = chatbot_response(user_msg)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
