import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

# Define constants
MOVES = ["rock", "paper", "scissors"]
MOVE_TO_IDX = {move: i for i, move in enumerate(MOVES)}
IDX_TO_MOVE = {i: move for i, move in enumerate(MOVES)}

# Counter move function
def get_counter_move(move):
    if move == "rock":
        return "paper"
    elif move == "paper":
        return "scissors"
    elif move == "scissors":
        return "rock"
    return random.choice(MOVES)

class TransformerModel(nn.Module):
    def __init__(self, input_size=3, d_model=64, nhead=4, num_layers=2, output_size=3, max_seq_length=20):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        
        # Create padding mask (to handle sequences of different lengths)
        # In this case, we'll assume all sequences are padded to the same length
        # so we don't need a mask, but in a real application, you would create one
        
        # Embed input
        x = self.embedding(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get output from the last position in sequence
        x = x[:, -1, :]
        
        # Final output layer
        output = self.fc_out(x)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]

class PatternDetector:
    """Simple rule-based pattern detector to handle repeating sequences"""
    def __init__(self, max_pattern_length=5):
        self.max_pattern_length = max_pattern_length
    
    def detect_pattern(self, moves):
        """Detect common patterns like alternating moves, repeating sequences, etc."""
        if len(moves) < 4:  # Need at least a few moves to detect patterns
            return None, 0.0
        
        # Check for alternating pattern (A, B, A, B, ...)
        if len(moves) >= 6:
            alternating = True
            for i in range(len(moves) - 4):
                if moves[i] != moves[i + 2] or moves[i + 1] != moves[i + 3]:
                    alternating = False
                    break
            
            if alternating:
                # If we have (A, B, A, B, A, B), predict A next
                return moves[-2], 0.9
        
        # Check for repeating patterns of length 2-5
        for pattern_len in range(2, min(self.max_pattern_length + 1, len(moves) // 2 + 1)):
            # Get the most recent pattern
            pattern = moves[-pattern_len:]
            
            # Check if this pattern repeats in the history
            repeats = 0
            for i in range(len(moves) - pattern_len, 0, -pattern_len):
                if i < pattern_len:
                    break
                
                if moves[i-pattern_len:i] == pattern:
                    repeats += 1
                else:
                    break
            
            if repeats >= 2:  # Pattern repeats at least twice
                # Predict the next move after the pattern
                return moves[pattern_len], 0.8 + (0.05 * repeats)  # Higher confidence with more repeats
        
        return None, 0.0

class TransformerStrategy:
    def __init__(self, sequence_length=20, learning_rate=0.001, memory_size=2000):
        """Initialize the Transformer strategy with training capabilities"""
        # Set up model parameters
        self.sequence_length = sequence_length
        self.history = []
        self.last_prediction = None
        
        # Pattern detector for explicit pattern recognition
        self.pattern_detector = PatternDetector()
        
        # Set up training components
        try:
            self.model = TransformerModel(max_seq_length=sequence_length)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            
            # Memory buffer for experience replay
            self.memory = deque(maxlen=memory_size)
            
            # Tracking metrics
            self.training_losses = []
            self.prediction_accuracy = []
            
            self.load_trained_weights_if_available()
            self.model.eval()  # Start in evaluation mode
            
            # Ensemble weights (how much to trust each component)
            self.transformer_weight = 0.7
            self.pattern_detector_weight = 0.3
            
        except Exception as e:
            print(f"Error initializing Transformer model: {e}")
            # Set up fallback random strategy
            self.model = None
            self.optimizer = None
            self.criterion = None
            self.memory = deque(maxlen=10)
            self.training_losses = []
            self.prediction_accuracy = []

    def counter_move(self, move):
        """Return the move that beats the given move"""
        return get_counter_move(move)

    def load_trained_weights_if_available(self):
        """Load pre-trained model weights if available"""
        try:
            if os.path.exists("transformer_rps_model.pt"):
                self.model.load_state_dict(torch.load("transformer_rps_model.pt"))
                print("Loaded pre-trained transformer model.")
                return True
            else:
                print("No pre-trained transformer model found. Starting with random weights.")
                return False
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return False

    def save_model(self):
        """Save the current model weights"""
        try:
            if self.model:
                torch.save(self.model.state_dict(), "transformer_rps_model.pt")
                print("Model saved to transformer_rps_model.pt")
                return True
            return False
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def encode_move(self, move):
        """One-hot encode a single move"""
        try:
            tensor = torch.zeros(1, 1, len(MOVES))
            tensor[0, 0, MOVE_TO_IDX[move]] = 1
            return tensor
        except Exception as e:
            print(f"Error encoding move: {e}")
            # Return a random encoding as fallback
            tensor = torch.zeros(1, 1, len(MOVES))
            tensor[0, 0, random.randint(0, len(MOVES)-1)] = 1
            return tensor
    
    def encode_sequence(self, sequence):
        """One-hot encode a sequence of moves"""
        try:
            if len(sequence) < self.sequence_length:
                # Pad with random moves if needed
                padding = [random.choice(MOVES) for _ in range(self.sequence_length - len(sequence))]
                sequence = padding + sequence
            
            # Use last n moves where n = sequence_length
            recent_sequence = sequence[-self.sequence_length:]
            
            # Create one-hot encoded tensor
            tensor = torch.zeros(1, self.sequence_length, len(MOVES))
            for i, move in enumerate(recent_sequence):
                tensor[0, i, MOVE_TO_IDX[move]] = 1
            
            return tensor
        except Exception as e:
            print(f"Error encoding sequence: {e}")
            # Return a random encoding as fallback
            tensor = torch.zeros(1, self.sequence_length, len(MOVES))
            for i in range(self.sequence_length):
                tensor[0, i, random.randint(0, len(MOVES)-1)] = 1
            return tensor
    
    def predict_next_move(self, player_history):
        """Predict the player's next move based on their history"""
        if not self.model or len(player_history) < 2:  # Need at least some history
            self.last_prediction = random.choice(MOVES)
            return self.last_prediction

        try:
            # First check for explicit patterns
            pattern_prediction, pattern_confidence = self.pattern_detector.detect_pattern(player_history)
            
            # Prepare input sequence for transformer
            inputs = self.encode_sequence(player_history)
            
            # Make transformer prediction
            with torch.no_grad():
                output = self.model(inputs)
                probs = F.softmax(output, dim=1)[0]
                
            # Get highest probability move and its confidence
            transformer_pred_idx = torch.argmax(probs).item()
            transformer_prediction = IDX_TO_MOVE[transformer_pred_idx]
            transformer_confidence = probs[transformer_pred_idx].item()
            
            # Ensemble decision
            if pattern_prediction and pattern_confidence > 0.6:
                # If pattern detector is very confident, use its prediction
                if pattern_confidence > 0.8:
                    self.last_prediction = pattern_prediction
                # Otherwise blend predictions
                else:
                    # If both predict the same move, use it
                    if pattern_prediction == transformer_prediction:
                        self.last_prediction = pattern_prediction
                    # Otherwise use weighted confidence to decide
                    else:
                        pattern_weight = self.pattern_detector_weight * pattern_confidence
                        transformer_weight = self.transformer_weight * transformer_confidence
                        
                        if pattern_weight > transformer_weight:
                            self.last_prediction = pattern_prediction
                        else:
                            self.last_prediction = transformer_prediction
            else:
                # No strong pattern detected, use transformer prediction
                self.last_prediction = transformer_prediction
            
            return self.last_prediction
        except Exception as e:
            print(f"Error predicting next move: {e}")
            self.last_prediction = random.choice(MOVES)
            return self.last_prediction

    def select_move(self, player_history):
        """Choose the best counter move based on prediction"""
        predicted_move = self.predict_next_move(player_history)
        return self.counter_move(predicted_move)
    
    def add_to_memory(self, state, target):
        """Add experience to replay memory"""
        try:
            self.memory.append((state, target))
        except Exception as e:
            print(f"Error adding to memory: {e}")
    
    def train_on_move(self, player_history, actual_next_move):
        """Train on a single move as it happens"""
        if not self.model or len(player_history) < self.sequence_length:
            return None  # Not enough data yet or model not available
        
        try:
            # Add to memory
            state = self.encode_sequence(player_history[:-1] if len(player_history) > 1 else player_history)
            target = MOVE_TO_IDX[actual_next_move]
            self.add_to_memory(state, target)
            
            # Evaluate prediction accuracy
            if self.last_prediction:
                correct = (self.last_prediction == actual_next_move)
                self.prediction_accuracy.append(1 if correct else 0)
                
                # Adjust ensemble weights based on which component was more accurate
                if correct:
                    # Keep weights as they are
                    pass
                else:
                    # Try to determine which component would have been correct
                    pattern_pred, _ = self.pattern_detector.detect_pattern(player_history[:-1])
                    
                    # If pattern detector was correct but overall prediction was wrong
                    if pattern_pred == actual_next_move:
                        # Increase pattern detector weight
                        self.pattern_detector_weight = min(0.5, self.pattern_detector_weight + 0.05)
                        self.transformer_weight = 1 - self.pattern_detector_weight
                    
                    # If transformer was wrong, adjust
                    with torch.no_grad():
                        inputs = self.encode_sequence(player_history[:-1])
                        output = self.model(inputs)
                        probs = F.softmax(output, dim=1)[0]
                        transformer_pred_idx = torch.argmax(probs).item()
                        transformer_pred = IDX_TO_MOVE[transformer_pred_idx]
                        
                        if transformer_pred != actual_next_move and pattern_pred != actual_next_move:
                            # Both were wrong, slightly favor transformer (which can learn)
                            self.transformer_weight = min(0.8, self.transformer_weight + 0.02)
                            self.pattern_detector_weight = 1 - self.transformer_weight
            
            # Only train if we have enough data
            if len(self.memory) < min(10, self.sequence_length):
                return None
                
            # Train on a batch from memory
            return self.train_batch()
        except Exception as e:
            print(f"Error training on move: {e}")
            return None
    
    def train_batch(self, batch_size=16):
        """Train on a random batch from memory"""
        if not self.model or len(self.memory) < batch_size:
            batch_size = len(self.memory) if self.model else 0
            
        if batch_size < 2:
            return None
            
        try:    
            # Sample from memory
            batch = random.sample(self.memory, batch_size)
            
            # Prepare batch
            states = torch.cat([state for state, _ in batch], dim=0)
            targets = torch.tensor([target for _, target in batch], dtype=torch.long)
            
            # Forward pass
            self.model.train()
            outputs = self.model(states)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store loss for metrics
            self.training_losses.append(loss.item())
            
            self.model.eval()  # Switch back to evaluation mode
            
            return loss.item()
        except Exception as e:
            print(f"Error training batch: {e}")
            return None
    
    def get_training_metrics(self):
        """Return training metrics for reporting"""
        try:
            metrics = {
                "avg_loss": np.mean(self.training_losses[-100:]) if self.training_losses else float('nan'),
                "avg_accuracy": np.mean(self.prediction_accuracy[-100:]) if self.prediction_accuracy else float('nan'),
                "total_samples": len(self.memory),
                "total_trained": len(self.training_losses),
                "transformer_weight": self.transformer_weight,
                "pattern_detector_weight": self.pattern_detector_weight
            }
            return metrics
        except Exception as e:
            print(f"Error getting training metrics: {e}")
            return {
                "avg_loss": 0.0,
                "avg_accuracy": 0.0,
                "total_samples": 0,
                "total_trained": 0,
                "transformer_weight": 0.7,
                "pattern_detector_weight": 0.3
            }