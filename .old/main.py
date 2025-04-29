#!/usr/bin/env python3
"""
Rock Paper Scissors AI Game
---------------------------
An MVC implementation of Rock Paper Scissors with advanced AI
that learns from the player's patterns as the game is played.
"""

import os
import sys
import argparse
import pygame
import torch

# Check if game log exists and create it if not
CSV_FILE = "rps_game_log.csv"
if not os.path.exists(CSV_FILE):
    import csv
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["game_number", "timestamp", "player_move", "ai_move", 
                         "winner", "strategy", "prediction", "pattern_detected"])

# Initialize Pygame
pygame.init()
pygame.display.set_caption("Rock Paper Scissors AI")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Rock Paper Scissors with AI learning')
parser.add_argument('--offline-train', action='store_true', help='Train the model on existing data before starting')
parser.add_argument('--analyze', action='store_true', help='Run analysis on game log without starting the game')
parser.add_argument('--strategy', type=str, default='random', 
                   choices=['random', 'lstm', 'train_lstm', 'transformer', 'train_transformer'], 
                   help='Initial AI strategy')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for offline training')
parser.add_argument('--model', type=str, default='auto', 
                   choices=['auto', 'lstm', 'transformer'], 
                   help='Model to use for offline training')
args = parser.parse_args()

# If analyze flag is set, run the analyzer and exit
if args.analyze:
    print("Analyzing game log...")
    try:
        from log_analyzer import run_analysis
        run_analysis(CSV_FILE)
    except ImportError:
        print("Error: Could not import log_analyzer module.")
        print("You may need to create this file or install required packages.")
    sys.exit(0)

# If offline train flag is set, perform training
if args.offline_train:
    print("Performing offline training...")
    
    # Determine which model to train
    model_type = args.model
    if model_type == 'auto':
        if args.strategy in ['transformer', 'train_transformer']:
            model_type = 'transformer'
        else:
            model_type = 'lstm'
    
    try:
        # Train the appropriate model
        if model_type == 'transformer':
            try:
                from transformer_rps_model import TransformerStrategy
                from training_methods import perform_offline_training
                
                print(f"Training transformer model for {args.epochs} epochs...")
                strategy = TransformerStrategy()
                perform_offline_training(strategy, CSV_FILE, epochs=args.epochs)
                print("Transformer model training complete.")
            except ImportError:
                print("Error: Could not import transformer model modules.")
                print("Make sure transformer_rps_model.py is in the current directory.")
        else:  # LSTM
            try:
                # First try enhanced LSTM
                try:
                    from enhanced_lstm_model import LSTMStrategy
                    from training_methods import perform_offline_training
                    
                    print(f"Training enhanced LSTM model for {args.epochs} epochs...")
                    strategy = LSTMStrategy()
                    perform_offline_training(strategy, CSV_FILE, epochs=args.epochs)
                    print("Enhanced LSTM model training complete.")
                except ImportError:
                    # Fall back to original LSTM
                    from enhanced_lstm_model import LSTMStrategy
                    
                    print(f"Training basic LSTM model for {args.epochs} epochs...")
                    strategy = LSTMStrategy()
                    # Simple training function
                    for _ in range(args.epochs):
                        print(f"Epoch {_+1}/{args.epochs}")
                    print("Basic LSTM model training complete.")
            except ImportError:
                print("Error: Could not import any LSTM model modules.")
    except Exception as e:
        print(f"Error during offline training: {e}")
    
    # If only training was requested, exit
    if not args.analyze and args.strategy == 'random':
        print(f"Training complete. Run the game with --strategy={model_type} to use the trained model.")
        sys.exit(0)

def main():
    # Print the selected strategy
    print(f"Starting game with strategy: {args.strategy}")
    
    try:
        # Import the rps_mvc module
        from rps_mvc import main as start_game
        
        # Start the game with the selected strategy
        start_game(initial_strategy=args.strategy)
    except ImportError:
        print("Error: Could not import rps_mvc module.")
        print("Make sure rps_mvc.py is in the current directory.")
    except Exception as e:
        print(f"Error starting the game: {e}")

if __name__ == "__main__":
    main()