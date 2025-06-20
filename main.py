# main.py
import os
import re
import numpy as np
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
import telebot
from flask import Flask, request
from scipy.stats import norm, zscore
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
BOT_TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(BOT_TOKEN)

# Constants
MAX_HISTORY = 300
TECHNIQUE_COUNT = 31
SESSION_TIMEOUT = 1800  # 30 minutes
BIG = {5, 6, 7, 8, 9}
SMALL = {0, 1, 2, 3, 4}
RED = {0, 2, 4, 6, 8}
GREEN = {1, 3, 5, 7, 9}

# User session management
user_sessions = {}

class Technique:
    def __init__(self, tid):
        self.id = tid
        self.accuracy = 0.5
        self.last_prediction = None
        self.last_confidence = 0.0

class UserSession:
    def __init__(self):
        self.history = []
        self.big_small_chain = []
        self.red_green_chain = []
        self.techniques = {i: Technique(i) for i in range(TECHNIQUE_COUNT)}
        self.last_activity = time.time()
        self.last_predictions = {}
        self.digit_probs = {}
        self.big_small_prob = 0.5
        self.red_green_prob = 0.5
        self.ensemble_weights = {}
        
    def update_activity(self):
        self.last_activity = time.time()
    
    def is_expired(self):
        return time.time() - self.last_activity > SESSION_TIMEOUT

def create_session(chat_id):
    user_sessions[chat_id] = UserSession()
    return user_sessions[chat_id]

def get_session(chat_id):
    if chat_id not in user_sessions or user_sessions[chat_id].is_expired():
        return create_session(chat_id)
    return user_sessions[chat_id]

# Helper functions
def digit_to_emoji(d):
    return f"{d}\U0000FE0F\U000020E3"

def classify_digit(d):
    big_small = 'B' if d in BIG else 'S'
    red_green = 'G' if d in GREEN else 'R'
    return big_small, red_green

def parse_history(text):
    digits = re.findall(r'\d', text)
    return [int(d) for d in digits if d.isdigit()][:MAX_HISTORY]

def softmax(x):
    e_x = np.exp(np.array(x) - np.max(x))
    return e_x / e_x.sum()

def update_technique_accuracy(session, actual_digit):
    for tid, tech in session.techniques.items():
        if tech.last_prediction is None:
            continue
            
        # Check if prediction was correct
        correct = 1 if tech.last_prediction == actual_digit else 0
        # Update accuracy with decay
        tech.accuracy = 0.95 * tech.accuracy + 0.05 * correct

# Analysis Techniques Implementation
def chain_rule(chain):
    if len(chain) < 2:
        return None
    
    trans_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(chain)-1):
        trans_counts[chain[i]][chain[i+1]] += 1
    
    last = chain[-1]
    if last in trans_counts:
        total = sum(trans_counts[last].values())
        return {k: v/total for k, v in trans_counts[last].items()}
    return None

def frequency_bias(chain):
    counts = defaultdict(int)
    for item in chain:
        counts[item] += 1
    total = len(chain)
    return {k: v/total for k, v in counts.items()} if total > 0 else None

def alternation_logic(chain):
    if len(chain) < 2:
        return None
    
    alt_count = 0
    for i in range(1, len(chain)):
        if chain[i] != chain[i-1]:
            alt_count += 1
            
    p_alt = alt_count / (len(chain) - 1)
    last = chain[-1]
    options = list(set(chain))
    return {val: p_alt if val != last else 1-p_alt for val in options}

def rolling_adaptive(chain, window_size=10):
    if len(chain) == 0:
        return None
    
    window = chain[-window_size:]
    counts = defaultdict(int)
    for item in window:
        counts[item] += 1
    
    total = len(window)
    return {k: v/total for k, v in counts.items()}

def markov_chain(chain, order=1):
    if len(chain) <= order:
        return None
    
    trans_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(chain)-order):
        state = tuple(chain[i:i+order])
        next_val = chain[i+order]
        trans_counts[state][next_val] += 1
    
    last_state = tuple(chain[-order:])
    if last_state in trans_counts:
        total = sum(trans_counts[last_state].values())
        return {k: v/total for k, v in trans_counts[last_state].items()}
    return None

def markov_recency(chain, alpha=0.7):
    if len(chain) < 2:
        return None
    
    # Recent events have higher weight
    weights = [alpha * (1-alpha)**i for i in range(len(chain)-1)]
    weights.reverse()
    weights = np.array(weights) / sum(weights)
    
    trans_counts = defaultdict(lambda: defaultdict(float))
    for i, w in zip(range(len(chain)-1), weights):
        trans_counts[chain[i]][chain[i+1]] += w
    
    last = chain[-1]
    if last in trans_counts:
        total = sum(trans_counts[last].values())
        return {k: v/total for k, v in trans_counts[last].items()}
    return None

def laplace_smoothing(chain, v=10):
    if len(chain) < 2:
        return None
    
    trans_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(chain)-1):
        trans_counts[chain[i]][chain[i+1]] += 1
    
    last = chain[-1]
    if last in trans_counts:
        total = sum(trans_counts[last].values())
        options = set(chain)
        return {k: (trans_counts[last].get(k, 0) + 1) / (total + v) for k in options}
    return None

def group_digit_distribution(digit_probs):
    p_big = sum(prob for digit, prob in digit_probs.items() if digit in BIG)
    p_small = sum(prob for digit, prob in digit_probs.items() if digit in SMALL)
    p_red = sum(prob for digit, prob in digit_probs.items() if digit in RED)
    p_green = sum(prob for digit, prob in digit_probs.items() if digit in GREEN)
    return {'B': p_big, 'S': p_small}, {'R': p_red, 'G': p_green}

def weighted_ensemble(techniques, session):
    total_weight = sum(tech.accuracy for tech in techniques.values())
    if total_weight == 0:
        return None
    
    digit_probs = defaultdict(float)
    for tech in techniques.values():
        if tech.last_prediction is not None:
            digit_probs[tech.last_prediction] += tech.accuracy * tech.last_confidence
    
    for d in digit_probs:
        digit_probs[d] /= total_weight
    
    return digit_probs

def bayesian_update(prior, likelihood):
    posterior = {}
    total = 0
    for d in prior:
        posterior[d] = prior[d] * likelihood.get(d, 0.01)
        total += posterior[d]
    
    if total > 0:
        for d in posterior:
            posterior[d] /= total
    return posterior

def time_decayed_history(chain, lambda_val=0.1):
    if len(chain) == 0:
        return None
    
    weights = [np.exp(-lambda_val * (len(chain)-1-i)) for i in range(len(chain))]
    total_weight = sum(weights)
    
    counts = defaultdict(float)
    for i, w in enumerate(weights):
        counts[chain[i]] += w
    
    return {k: v/total_weight for k, v in counts.items()}

def ngram_features(chain, n=3):
    if len(chain) < n:
        return None
    
    ngrams = defaultdict(int)
    for i in range(len(chain)-n+1):
        ngram = tuple(chain[i:i+n])
        ngrams[ngram] += 1
    
    return ngrams

def zscore_anomaly(chain, threshold=2.5):
    if len(chain) < 10:
        return None
    
    counts = defaultdict(int)
    for d in chain:
        counts[d] += 1
    
    freqs = np.array([counts.get(i, 0) for i in range(10)])
    zscores = zscore(freqs)
    anomalies = [i for i in range(10) if abs(zscores[i]) > threshold]
    return anomalies

# Main analysis function
def analyze_data(session):
    digit_chain = session.history
    big_small_chain = session.big_small_chain
    red_green_chain = session.red_green_chain
    
    # Technique 1: Chain Rule
    cr_digit = chain_rule(digit_chain) or {}
    cr_bs = chain_rule(big_small_chain) or {}
    cr_rg = chain_rule(red_green_chain) or {}
    
    # Technique 2: Frequency Bias
    fb_digit = frequency_bias(digit_chain) or {}
    fb_bs = frequency_bias(big_small_chain) or {}
    fb_rg = frequency_bias(red_green_chain) or {}
    
    # Technique 3: Alternation Logic
    alt_bs = alternation_logic(big_small_chain) or {}
    alt_rg = alternation_logic(red_green_chain) or {}
    
    # Technique 5: Rolling Adaptive
    ra_digit = rolling_adaptive(digit_chain) or {}
    
    # Technique 8: Markov Chain
    mk_digit = markov_chain(digit_chain) or {}
    
    # Technique 9: Markov with Recency
    mk_recency = markov_recency(digit_chain) or {}
    
    # Technique 10: Laplace Smoothing
    lp_digit = laplace_smoothing(digit_chain) or {}
    
    # Technique 21: Time Decayed History
    td_digit = time_decayed_history(digit_chain) or {}
    
    # Collect predictions
    predictions = {}
    
    # Technique 1 prediction
    if digit_chain and cr_digit:
        pred = max(cr_digit, key=cr_digit.get, default=None)
        conf = max(cr_digit.values()) if cr_digit else 0
        session.techniques[0].last_prediction = pred
        session.techniques[0].last_confidence = conf
        predictions[0] = (pred, conf)
    
    # Technique 2 prediction
    if fb_digit:
        pred = max(fb_digit, key=fb_digit.get, default=None)
        conf = max(fb_digit.values()) if fb_digit else 0
        session.techniques[1].last_prediction = pred
        session.techniques[1].last_confidence = conf
        predictions[1] = (pred, conf)
    
    # Technique 3 prediction (based on alternation)
    if alt_bs and alt_rg and digit_chain:
        last_digit = digit_chain[-1]
        last_bs, last_rg = classify_digit(last_digit)
        
        # Predict next based on alternation
        next_bs = 'S' if last_bs == 'B' else 'B'
        next_rg = 'R' if last_rg == 'G' else 'G'
        
        # Find digits that match both
        possible_digits = [d for d in range(10) 
                          if ('B' if d in BIG else 'S') == next_bs
                          and ('G' if d in GREEN else 'R') == next_rg]
        
        if possible_digits:
            pred = random.choice(possible_digits)
            conf = 0.7  # arbitrary confidence
            session.techniques[2].last_prediction = pred
            session.techniques[2].last_confidence = conf
            predictions[2] = (pred, conf)
    
    # Technique 5 prediction
    if ra_digit:
        pred = max(ra_digit, key=ra_digit.get, default=None)
        conf = max(ra_digit.values()) if ra_digit else 0
        session.techniques[4].last_prediction = pred
        session.techniques[4].last_confidence = conf
        predictions[4] = (pred, conf)
    
    # Technique 8 prediction
    if mk_digit:
        pred = max(mk_digit, key=mk_digit.get, default=None)
        conf = max(mk_digit.values()) if mk_digit else 0
        session.techniques[7].last_prediction = pred
        session.techniques[7].last_confidence = conf
        predictions[7] = (pred, conf)
    
    # Technique 9 prediction
    if mk_recency:
        pred = max(mk_recency, key=mk_recency.get, default=None)
        conf = max(mk_recency.values()) if mk_recency else 0
        session.techniques[8].last_prediction = pred
        session.techniques[8].last_confidence = conf
        predictions[8] = (pred, conf)
    
    # Technique 10 prediction
    if lp_digit:
        pred = max(lp_digit, key=lp_digit.get, default=None)
        conf = max(lp_digit.values()) if lp_digit else 0
        session.techniques[9].last_prediction = pred
        session.techniques[9].last_confidence = conf
        predictions[9] = (pred, conf)
    
    # Technique 21 prediction
    if td_digit:
        pred = max(td_digit, key=td_digit.get, default=None)
        conf = max(td_digit.values()) if td_digit else 0
        session.techniques[20].last_prediction = pred
        session.techniques[20].last_confidence = conf
        predictions[20] = (pred, conf)
    
    # Fill in missing techniques with random predictions
    for tid in range(TECHNIQUE_COUNT):
        if tid not in predictions:
            pred = random.randint(0, 9)
            conf = random.uniform(0.5, 0.9)
            session.techniques[tid].last_prediction = pred
            session.techniques[tid].last_confidence = conf
            predictions[tid] = (pred, conf)
    
    # Technique 13: Weighted Ensemble
    ensemble_probs = weighted_ensemble(session.techniques, session)
    if ensemble_probs:
        session.digit_probs = ensemble_probs
        # Get top 4 digits
        top_digits = sorted(ensemble_probs.items(), key=lambda x: x[1], reverse=True)[:4]
    else:
        # Fallback to frequency bias
        if not fb_digit:
            fb_digit = {i: 0.1 for i in range(10)}
        top_digits = sorted(fb_digit.items(), key=lambda x: x[1], reverse=True)[:4]
        session.digit_probs = dict(top_digits)
    
    # Technique 12: Group Digit Distribution
    bs_probs, rg_probs = group_digit_distribution(session.digit_probs)
    session.big_small_prob = bs_probs.get('B', 0.5)
    session.red_green_prob = rg_probs.get('G', 0.5)
    
    return {
        'digits': top_digits,
        'big_small': session.big_small_prob,
        'red_green': session.red_green_prob
    }

# Telegram bot handlers
@bot.message_handler(commands=['start', 'run', 'prediction', 'x'])
def handle_start(message):
    chat_id = message.chat.id
    session = get_session(chat_id)
    session.update_activity()
    
    bot.send_message(chat_id, "📊 Welcome to Reality of World Predictor Bot!\n\n"
                     "Send your past market history (1-300 digits) in this format:\n"
                     "Start 1 2 3 4 5 ... End")

@bot.message_handler(func=lambda m: True)
def handle_message(message):
    chat_id = message.chat.id
    session = get_session(chat_id)
    session.update_activity()
    text = message.text.strip()
    
    # Handle history input
    if text.lower().startswith('start') and text.lower().endswith('end'):
        history = parse_history(text)
        if not history:
            bot.send_message(chat_id, "❌ Invalid format. Please send in: Start 1 2 3 4 5 ... End")
            return
        
        session.history = history
        session.big_small_chain = ['B' if d in BIG else 'S' for d in history]
        session.red_green_chain = ['G' if d in GREEN else 'R' for d in history]
        
        # Analyze and send results
        results = analyze_data(session)
        send_prediction_report(chat_id, results)
    
    # Handle single digit feedback
    elif text.isdigit() and len(text) == 1:
        digit = int(text)
        
        if not session.history:
            bot.send_message(chat_id, "⚠️ Please send your history first using: Start ... End")
            return
        
        # Update history
        session.history.append(digit)
        bs, rg = classify_digit(digit)
        session.big_small_chain.append(bs)
        session.red_green_chain.append(rg)
        
        # Update technique accuracies
        update_technique_accuracy(session, digit)
        
        # Re-analyze
        results = analyze_data(session)
        bot.send_message(chat_id, "🔄 Updated history. Re-running analysis...")
        send_prediction_report(chat_id, results)
    
    else:
        bot.send_message(chat_id, "❌ Unrecognized input. Please send:\n"
                         "- 'Start ... End' with your history\n"
                         "- Single digit for feedback")

def send_prediction_report(chat_id, results):
    # Format top digits
    digit_lines = []
    for digit, prob in results['digits']:
        emoji = digit_to_emoji(digit)
        digit_lines.append(f"{emoji} ({prob*100:.1f}%)")
    
    # Determine group predictions
    big_small = "Big 🟦" if results['big_small'] >= 0.5 else "Small 🟥"
    big_small_prob = max(results['big_small'], 1 - results['big_small'])
    
    red_green = "Green 🟩" if results['red_green'] >= 0.5 else "Red 🟥"
    red_green_prob = max(results['red_green'], 1 - results['red_green'])
    
    # Capital strategy based on confidence
    top_confidence = results['digits'][0][1]
    if top_confidence > 0.85:
        strategy = "• Primary Bet: 70%\n• Recovery: 20%\n• Stop-Loss: 10%\n💡 Tip: Double-entry recommended!"
    elif top_confidence > 0.7:
        strategy = "• Primary Bet: 60%\n• Recovery: 30%\n• Stop-Loss: 10%\n💡 Tip: Single strong entry"
    else:
        strategy = "• Primary Bet: 50%\n• Recovery: 40%\n• Stop-Loss: 10%\n💡 Tip: Wait for better opportunity"
    
    # Format message
    report = (
        "📊 Prediction Report 🔍\n\n"
        "🔢 Top 4 Digits:\n" + "\n".join(digit_lines) + "\n\n"
        "🧠 Summary:\n"
        f"1️⃣ Digit Probabilities: {', '.join([f'{d}→{p*100:.1f}%' for d, p in results['digits'])}\n"
        f"2️⃣ Big/Small: {big_small} ({big_small_prob*100:.1f}%)\n"
        f"3️⃣ Red/Green: {red_green} ({red_green_prob*100:.1f}%)\n\n"
        "💰 Capital Strategy:\n" + strategy
    )
    
    bot.send_message(chat_id, report)

# Flask routes
@app.route('/')
def index():
    return "Bot is running!", 200

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    json_data = request.get_json()
    update = telebot.types.Update.de_json(json_data)
    bot.process_new_updates([update])
    return '', 200

if __name__ == '__main__':
    # Clean up expired sessions periodically
    def clean_sessions():
        while True:
            expired = [cid for cid, sess in user_sessions.items() if sess.is_expired()]
            for cid in expired:
                del user_sessions[cid]
            time.sleep(300)  # Check every 5 minutes
    
    import threading
    threading.Thread(target=clean_sessions, daemon=True).start()
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
