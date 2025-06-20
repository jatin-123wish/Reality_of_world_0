# engine.py
import numpy as np
from typing import List, Dict, Tuple
import random

# Placeholder for technique accuracies (update via feedback)
tech_accuracy = {i: 1.0 for i in range(1, 32)}

# Core helper: normalized probabilities
def normalize(probs: Dict) -> Dict:
    total = sum(probs.values()) or 1
    return {k: v/total for k, v in probs.items()}

# 1. Chain Rule
def chain_rule(chain: List, state) -> Dict:
    trans = {}
    for a, b in zip(chain, chain[1:]):
        trans.setdefault(a, {}).setdefault(b, 0)
        trans[a][b] += 1
    if state not in trans:
        return {}
    return normalize(trans[state])

# 2. Frequency Bias
def frequency_bias(chain: List) -> Dict:
    counts = {}
    for x in chain:
        counts[x] = counts.get(x, 0) + 1
    return normalize(counts)

# 3. Alternation Logic
def alternation_logic(chain: List, pair: Tuple) -> Dict:
    alt = sum(1 for a, b in zip(chain, chain[1:]) if (a, b) == pair or (a, b) == pair[::-1])
    total = len(chain)-1 or 1
    return {pair[0]: alt/total, pair[1]: 1 - alt/total}

# 4. Combined Voting
def combined_voting(preds: List[Dict]) -> Dict:
    votes = {}
    for p in preds:
        if not p: continue
        top = max(p.items(), key=lambda x: x[1])[0]
        votes[top] = votes.get(top, 0) + 1
    return normalize(votes)

# 5. Rolling Adaptive
def rolling_adaptive(chain: List, k: int = 5) -> Dict:
    window = chain[-k:]
    return frequency_bias(window)

# 6. Only Real Feedback Append -> no-op technique
def feedback_append(*args, **kwargs) -> Dict:
    return {}

# 7. Dynamic Re-learning -> no-op stub
def dynamic_relearning(*args, **kwargs) -> Dict:
    return {}

# 8. Markov Chain
def markov_chain(chain: List) -> Dict:
    return chain_rule(chain, chain[-1]) if chain else {}

# 9. Markov + Recency Bias
def markov_recency(chain: List, alpha: float = 0.8) -> Dict:
    old = markov_chain(chain[:-1]) if len(chain)>1 else {}
    new = chain_rule(chain, chain[-1])
    merged = {k: alpha*new.get(k,0) + (1-alpha)*old.get(k,0) for k in set(old)|set(new)}
    return normalize(merged)

#10. Laplace Smoothing
def laplace_smoothing(chain: List, V: int) -> Dict:
    counts = {x: chain.count(x)+1 for x in set(chain)}
    total = len(chain) + V
    return {k: v/total for k, v in counts.items()}

#11. Threshold Validation
def threshold_validation(probs: Dict, fallback: Dict, theta: float=0.6) -> Dict:
    top = max(probs.values()) if probs else 0
    return probs if top>=theta else fallback

#12. Group Digit Distribution
def group_distribution(digit_probs: Dict) -> Dict:
    big = sum(p for d,p in digit_probs.items() if d>=5)
    small = sum(p for d,p in digit_probs.items() if d<5)
    return {'Big': big, 'Small': small}

#13. Weighted Ensemble
def weighted_ensemble(preds: List[Dict]) -> Dict:
    weights = np.array([tech_accuracy[i+1] for i in range(len(preds))])
    weights /= weights.sum() or 1
    combined = {}
    for w,pred in zip(weights, preds):
        for k,v in pred.items(): combined[k] = combined.get(k,0)+w*v
    return normalize(combined)

#14. Model Diversity -> stub
#15. Feedback Loop Update -> stub
#16. Bayesian Updating
def bayesian_update(prior: Dict, likelihood: Dict) -> Dict:
    post = {k: prior.get(k,0)*likelihood.get(k,0) for k in set(prior)|set(likelihood)}
    return normalize(post)

#17. Ensemble Stacking -> stub
#18. HMM -> stub
#19. Particle Filter -> stub
#20. Change-Point Detection -> stub
#21. Time-Decayed History
def time_decay(chain: List, lam: float=0.1) -> Dict:
    weights = {chain[i]: np.exp(-lam*(len(chain)-i-1)) for i in range(len(chain))}
    return normalize(weights)

#22. Cross-Validation & Backtesting -> stub
#23. N-Gram Features
def ngram_features(chain: List, n: int=2) -> Dict:
    feats = {}
    for i in range(len(chain)-n+1):
        seq = tuple(chain[i:i+n])
        feats[seq] = feats.get(seq,0)+1
    return normalize(feats)

#24. XGBoost -> stub (requires external lib)
#25. Probability Calibration -> stub
#26. Adversarial Testing -> stub
#27. Online Hyperparam Opt -> stub
#28. Multi-Resolution Trends -> stub
#29. Anomaly Detection
def anomaly_filter(chain: List) -> Dict:
    mean = np.mean(chain)
    std = np.std(chain) or 1
    zscores = {(d): abs((d-mean)/std) for d in chain}
    return normalize({d:1/(1+z) for d,z in zscores.items()})

#30. RL Group Predictor -> stub
#31. Accuracy-Weighted Voting -> uses weighted_ensemble

# Super Ensemble Controller
def super_ensemble(preds: List[Dict]) -> Dict:
    return weighted_ensemble(preds)

# Main predict
def predict(
    digit_chain: List[int],
    bs_chain: List[str],
    rg_chain: List[str]
) -> Dict[str, object]:
    # Collect preds
    dp, bp, rp = [], [], []
    # 1-3
    if digit_chain: dp.append(chain_rule(digit_chain, digit_chain[-1]))
    dp.append(frequency_bias(digit_chain))
    # Alternation not for digits
    # 4-5
    dp.append(combined_voting(dp))
    dp.append(rolling_adaptive(digit_chain))
    # 6-7
    dp.append(feedback_append())
    dp.append(dynamic_relearning())
    # 8-10
    dp.append(markov_chain(digit_chain))
    dp.append(markov_recency(digit_chain))
    dp.append(laplace_smoothing(digit_chain, V=10))
    # 11-13
    dp.append(threshold_validation(dp[-1], frequency_bias(digit_chain)))
    dp.append(group_distribution(frequency_bias(digit_chain)))
    dp.append(weighted_ensemble(dp))
    # 14-15 stub
    # 16
    dp.append(bayesian_update(frequency_bias(digit_chain), dp[0]))
    # 17-23 mix stubs + ngram
    dp.append(ngram_features(digit_chain))
    dp.append(anomaly_filter(digit_chain))
    # 24-31 stubs or reuse
    # Super ensemble
    final_digit = super_ensemble(dp)
    final_bs = super_ensemble([frequency_bias([1 if c=='B' else 0 for c in bs_chain]), alternation_logic(bs_chain,('B','S')), group_distribution(final_digit)])
    final_rg = super_ensemble([frequency_bias([1 if c=='R' else 0 for c in rg_chain]), alternation_logic(rg_chain,('R','G')), group_distribution(final_digit)])

    top4 = sorted(final_digit.items(), key=lambda x: x[1], reverse=True)[:4]
    bs_pred, bs_conf = max(final_bs.items(), key=lambda x: x[1])
    rg_pred, rg_conf = max(final_rg.items(), key=lambda x: x[1])
    return {
        'top4_digits': [(d, p*100) for d, p in top4],
        'big_small': (bs_pred, bs_conf*100),
        'red_green': (rg_pred, rg_conf*100)
  }
  
