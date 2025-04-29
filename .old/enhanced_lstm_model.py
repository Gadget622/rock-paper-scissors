import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import os

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

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class LSTMStrategy:
    def __init__(self, sequence_length=10, learning_rate=0.001, memory_size=1000):
        """Initialize the LSTM strategy with training capabilities"""
        # Set up model parameters
        self.sequence_length = sequence_length
        self.history = []
        self.last_prediction = None
        
        # Set up training components
        try:
            self.model = LSTMModel()
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()
            
            # Memory buffer for experience replay
            self.memory = deque(maxlen=memory_size)
            
            # Tracking metrics
            self.training_losses = []
            self.prediction_accuracy = []
            
            self.load_trained_weights_if_available()
            self.model.eval()  # Start in evaluation mode
        except Exception as e:
            print(f"Error initializing LSTM model: {e}")
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
            if os.path.exists("lstm_rps_model.pt"):
                self.model.load_state_dict(torch.load("lstm_rps_model.pt"))
                print("Loaded pre-trained model.")
                return True
            else:
                print("No pre-trained model found. Starting with random weights.")
                return False
        except Exception as e:
            print(f"Error loading model weights: {e}")
            return False

    def save_model(self):
        """Save the current model weights"""
        try:
            if self.model:
                torch.save(self.model.state_dict(), "lstm_rps_model.pt")
                print("Model saved to lstm_rps_model.pt")
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
            # Prepare input sequence
            inputs = self.encode_sequence(player_history)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(inputs)
                
            prediction_idx = torch.argmax(F.softmax(output, dim=1)).item()
            self.last_prediction = IDX_TO_MOVE[prediction_idx]
            
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
            
            # Only train if we have enough data
            if len(self.memory) < min(10, self.sequence_length):
                return None
                
            # Train on a batch from memory
            return self.train_batch()
        except Exception as e:
            print(f"Error training on move: {e}")
            return None
    
    def train_batch(self, batch_size=8):
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
                "total_trained": len(self.training_losses)
            }
            return metrics
        except Exception as e:
            print(f"Error getting training metrics: {e}")
            return {
                "avg_loss": 0.0,
                "avg_accuracy": 0.0,
                "total_samples": 0,
                "total_trained": 0
            }