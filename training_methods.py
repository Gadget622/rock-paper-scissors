"""
Training methods for Rock Paper Scissors AI models
"""

import os
import random
import csv
import pandas as pd
import torch
import numpy as np
from collections import deque

def perform_offline_training(model, csv_file="rps_game_log.csv", epochs=10, batch_size=32, verbose=True):
    """
    Perform offline training on historical game data
    
    Args:
        model: The model to train (LSTMStrategy or TransformerStrategy)
        csv_file: Path to the CSV file with game data
        epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Whether to print progress
        
    Returns:
        Boolean indicating success/failure
    """
    if not os.path.exists(csv_file):
        if verbose:
            print(f"File not found: {csv_file}")
        return False
    
    try:
        # Load data
        if verbose:
            print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Extract player moves
        player_moves = df["player_move"].tolist()
        if len(player_moves) < 10:
            if verbose:
                print("Not enough data for training (need at least 10 moves)")
            return False
        
        if verbose:
            print(f"Loaded {len(player_moves)} moves")
            print("Creating training data...")
        
        # Create training data
        training_data = []
        sequence_length = getattr(model, 'sequence_length', 10)
        
        for i in range(len(player_moves) - sequence_length):
            # Input: sequence of moves
            sequence = player_moves[i:i+sequence_length]
            # Target: next move
            target = player_moves[i+sequence_length]
            
            # Encode sequence (handle different model types)
            try:
                if hasattr(model, 'encode_sequence'):
                    encoded_sequence = model.encode_sequence(sequence)
                    target_idx = get_move_idx(target)
                    training_data.append((encoded_sequence, target_idx))
            except Exception as e:
                if verbose:
                    print(f"Error encoding sequence: {e}")
        
        if verbose:
            print(f"Created {len(training_data)} training samples")
        
        # Train for specified epochs
        for epoch in range(epochs):
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
            
            # Shuffle data
            random.shuffle(training_data)
            
            # Train in batches
            total_loss = 0.0
            total_batches = 0
            
            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]
                if len(batch) < 2:
                    continue
                
                # Add batch to memory
                for sample in batch:
                    if hasattr(model, 'add_to_memory'):
                        model.add_to_memory(*sample)
                
                # Train on batch
                if hasattr(model, 'train_batch'):
                    loss = model.train_batch(batch_size=len(batch))
                    if loss is not None:
                        total_loss += loss
                        total_batches += 1
            
            # Epoch results
            if total_batches > 0 and verbose:
                avg_loss = total_loss / total_batches
                print(f"  Loss: {avg_loss:.4f}")
        
        # Save model
        if hasattr(model, 'save_model'):
            model.save_model()
            if verbose:
                print("Model saved")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"Error during offline training: {e}")
        return False

def get_move_idx(move):
    """Convert move string to index"""
    moves = ["rock", "paper", "scissors"]
    move_to_idx = {move: i for i, move in enumerate(moves)}
    return move_to_idx.get(move, 0)

def analyze_game_patterns(csv_file="rps_game_log.csv"):
    """
    Analyze patterns in the game history
    
    Args:
        csv_file: Path to the CSV file with game data
        
    Returns:
        Dictionary with pattern analysis
    """
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        
        # Extract sequences
        player_moves = df["player_move"].tolist()
        ai_moves = df["ai_move"].tolist()
        results = df["winner"].tolist()
        
        # Count move frequencies
        player_freqs = {}
        for move in player_moves:
            player_freqs[move] = player_freqs.get(move, 0) + 1
            
        # Look for patterns in player moves
        patterns = {}
        for n in range(2, 5):  # Check for patterns of length 2-4
            for i in range(len(player_moves) - n):
                pattern = tuple(player_moves[i:i+n])
                if pattern not in patterns:
                    patterns[pattern] = {'count': 0, 'next_moves': {}}
                patterns[pattern]['count'] += 1
                
                # Record next move if available
                if i + n < len(player_moves):
                    next_move = player_moves[i+n]
                    patterns[pattern]['next_moves'][next_move] = patterns[pattern]['next_moves'].get(next_move, 0) + 1
        
        # Calculate win rates
        total_games = len(results)
        if total_games == 0:
            return None
            
        win_rates = {}
        for result in results:
            win_rates[result] = win_rates.get(result, 0) + 1
            
        for result, count in win_rates.items():
            win_rates[result] = count / total_games
        
        # Analyze strategies' performance
        if 'strategy' in df.columns:
            strategies = {}
            for strategy in df['strategy'].unique():
                strategy_df = df[df['strategy'] == strategy]
                strategy_results = strategy_df["winner"].tolist()
                if not strategy_results:
                    continue
                    
                strategy_win_rates = {}
                for result in strategy_results:
                    strategy_win_rates[result] = strategy_win_rates.get(result, 0) + 1
                
                for result, count in strategy_win_rates.items():
                    strategy_win_rates[result] = count / len(strategy_results)
                    
                strategies[strategy] = strategy_win_rates
        else:
            strategies = None
        
        # Analyze prediction accuracy if available
        prediction_accuracy = None
        if 'prediction' in df.columns:
            # Get rows with valid predictions
            df_pred = df.dropna(subset=['prediction'])
            
            if not df_pred.empty:
                # Get pairs of prediction and next move
                predictions = df_pred['prediction'].tolist()
                next_moves = []
                
                for i in range(len(df_pred) - 1):
                    next_moves.append(df_pred.iloc[i+1]['player_move'])
                
                # Calculate accuracy
                correct = 0
                for pred, actual in zip(predictions[:-1], next_moves):
                    if pred == actual:
                        correct += 1
                
                if len(next_moves) > 0:
                    prediction_accuracy = correct / len(next_moves)
        
        # Return all analysis
        return {
            'total_games': total_games,
            'player_freqs': player_freqs,
            'win_rates': win_rates,
            'strategies': strategies,
            'patterns': patterns,
            'prediction_accuracy': prediction_accuracy
        }
    except Exception as e:
        print(f"Error analyzing game patterns: {e}")
        return None


def create_adaptive_ensemble(models):
    """
    Create an ensemble of multiple models with adaptive weighting
    
    Args:
        models: List of (model, weight) tuples
        
    Returns:
        Ensemble model
    """
    class AdaptiveEnsemble:
        def __init__(self, models):
            self.models = models  # List of (model, weight) tuples
            self.history = []
            self.last_prediction = None
            self.accuracy_history = []
            
        def predict_next_move(self, player_history):
            if not player_history:
                self.last_prediction = random.choice(["rock", "paper", "scissors"])
                return self.last_prediction
                
            # Get predictions from all models
            predictions = []
            for model, weight in self.models:
                try:
                    if hasattr(model, 'predict_next_move'):
                        pred = model.predict_next_move(player_history)
                        predictions.append((pred, weight))
                except Exception as e:
                    print(f"Error getting prediction from model: {e}")
            
            if not predictions:
                self.last_prediction = random.choice(["rock", "paper", "scissors"])
                return self.last_prediction
                
            # Weight the predictions
            vote_counts = {}
            for pred, weight in predictions:
                vote_counts[pred] = vote_counts.get(pred, 0) + weight
                
            # Choose the prediction with highest weight
            self.last_prediction = max(vote_counts.items(), key=lambda x: x[1])[0]
            return self.last_prediction
            
        def train_on_move(self, player_history, actual_next_move):
            # Train all component models
            for model, _ in self.models:
                try:
                    if hasattr(model, 'train_on_move'):
                        model.train_on_move(player_history, actual_next_move)
                except Exception as e:
                    print(f"Error training model: {e}")
                    
            # Adjust weights based on which models were correct
            if self.last_prediction is not None:
                for i, (model, weight) in enumerate(self.models):
                    if hasattr(model, 'last_prediction') and model.last_prediction is not None:
                        # Increase weight if correct, decrease if wrong
                        correct = (model.last_prediction == actual_next_move)
                        if correct:
                            self.models[i] = (model, min(1.0, weight + 0.05))
                        else:
                            self.models[i] = (model, max(0.1, weight - 0.02))
                
                # Normalize weights to sum to 1.0
                total = sum(weight for _, weight in self.models)
                if total > 0:
                    self.models = [(model, weight/total) for model, weight in self.models]
                
                # Track accuracy
                ensemble_correct = (self.last_prediction == actual_next_move)
                self.accuracy_history.append(1 if ensemble_correct else 0)
            
            return None  # No loss to return for now
            
        def get_training_metrics(self):
            metrics = {
                "avg_accuracy": np.mean(self.accuracy_history[-100:]) if self.accuracy_history else 0.0,
                "model_weights": {f"model_{i}": weight for i, (_, weight) in enumerate(self.models)}
            }
            return metrics
            
        def counter_move(self, move):
            if move == "rock":
                return "paper"
            elif move == "paper":
                return "scissors"
            elif move == "scissors":
                return "rock"
            return random.choice(["rock", "paper", "scissors"])
            
        def select_move(self, player_history):
            predicted_move = self.predict_next_move(player_history)
            return self.counter_move(predicted_move)
            
        def save_model(self):
            # Save each component model if they have save_model method
            for model, _ in self.models:
                if hasattr(model, 'save_model'):
                    model.save_model()
            return True
    
    return AdaptiveEnsemble(models)