import nltk
import string
import re
import random
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Sample training data with labeled intents
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
    """Text preprocessing: lowercase, tokenize, remove punctuation/stopwords, lemmatize."""
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [lemmatizer.lemmatize(t) for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(filtered)

# Preprocess all texts and extract labels
texts = [preprocess_text(sentence) for sentence, intent in training_data]
labels = [intent for sentence, intent in training_data]

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train Logistic Regression classifier for intent prediction
model = LogisticRegression()
model.fit(X, labels)

def extract_entities(text):
    """Simple regex based entity extraction for emails and phone numbers."""
    email = re.findall(r'\S+@\S+', text)
    phone = re.findall(r'\b\d{10}\b', text)
    return {"email": email, "phone": phone}

# Predefined responses keyed by intent
responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?"],
    "farewell": ["Goodbye! Have a great day.", "See you later!"],
    "thanks": ["You're welcome!", "Happy to help!"],
    "help": ["Sure, I can help! Please ask your question."],
    "product_info": ["We offer a range of products tailored to your needs.", "Please specify which product you'd like to know more about."],
    "hours": ["We are open from 9 AM to 6 PM, Monday to Friday."],
}

def chatbot_response(user_text):
    """Generate chatbot response based on user input."""
    processed = preprocess_text(user_text)
    vector = vectorizer.transform([processed])
    intent = model.predict(vector)[0]
    entity_info = extract_entities(user_text)

    response = random.choice(responses.get(intent, ["Sorry, I didn't understand that. Can you please rephrase?"]))

    # Append info if email or phone detected
    if entity_info["email"]:
        response += f" I noticed you mentioned an email address: {entity_info['email'][0]}"
    if entity_info["phone"]:
        response += f" I noticed you mentioned a phone number: {entity_info['phone'][0]}"

    return response

if __name__ == "__main__":
    print("Chatbot started! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(user_input))
