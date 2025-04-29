import random
from collections import Counter
from difflib import SequenceMatcher

# Constants
MOVES = ["rock", "paper", "scissors"]

# Sample player history for simulation/testing
opponent_moves = ["rock", "paper", "rock", "rock", "scissors", "paper", "rock"]
predicted_moves = []

# Naive Predictor: Predicts most common move so far
def predict_next_move(history):
    if not history:
        return random.choice(MOVES)
    freq = Counter(history)
    return freq.most_common(1)[0][0]

# Top-N Accuracy
def top_n_accuracy(predictions, actuals, n=1):
    correct = 0
    for pred, actual in zip(predictions, actuals):
        if isinstance(pred, list):
            if actual in pred[:n]:
                correct += 1
        else:
            if pred == actual:
                correct += 1
    return correct / len(actuals)

# Token Overlap Score (order-free)
def token_overlap(predictions, actuals):
    overlap = set(predictions) & set(actuals)
    return len(overlap) / len(set(actuals))

# Edit Distance (normalized)
def normalized_edit_distance(predictions, actuals):
    matcher = SequenceMatcher(None, predictions, actuals)
    ratio = matcher.ratio()  # 0.0 to 1.0 similarity
    return 1.0 - ratio  # Lower is better

# Run prediction
for i in range(len(opponent_moves) - 1):
    history = opponent_moves[:i+1]
    prediction = predict_next_move(history)
    predicted_moves.append(prediction)

# Evaluation
actual_next_moves = opponent_moves[1:]  # shift left to align with predictions

print("Predictions:", predicted_moves)
print("Actuals:    ", actual_next_moves)

print("\nEvaluation Metrics:")
print(f"Top-1 Accuracy:        {top_n_accuracy(predicted_moves, actual_next_moves):.2f}")
print(f"Token Overlap Score:   {token_overlap(predicted_moves, actual_next_moves):.2f}")
print(f"Normalized Edit Dist.: {normalized_edit_distance(predicted_moves, actual_next_moves):.2f}")
