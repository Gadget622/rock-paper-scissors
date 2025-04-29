# Rock Paper Scissors AI

A Rock Paper Scissors game with an LSTM neural network that learns from the player's patterns as the game is played.

## Overview

This project implements a Rock Paper Scissors game with several AI strategies:
- **Random**: Makes random moves (baseline)
- **LSTM**: Uses a pre-trained LSTM model to predict the player's next move
- **Train LSTM**: Uses LSTM predictions while continuously training on the player's moves

The game follows an MVC (Model-View-Controller) architecture and includes real-time visualization of the AI's learning progress.

## Features

- **Real-time Learning**: The LSTM model trains as you play, adapting to your patterns
- **Game Logging**: All games are logged to CSV for later analysis
- **Pattern Analysis**: Tools to analyze player patterns and AI prediction accuracy
- **Offline Training**: Train the model on historical game data without playing
- **Graphical Interface**: Visual feedback on game statistics and AI learning progress

## Files

- `rps_mvc.py`: Main game implementation with MVC architecture
- `enhanced_lstm_model.py`: LSTM neural network model with online training capability
- `training_methods.py`: Advanced training methods including data augmentation and validation
- `log_analyzer.py`: Tools to analyze game logs and visualize player patterns
- `main.py`: Entry point script with command-line options

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- Pandas
- Matplotlib
- NumPy
- Seaborn

Install dependencies:
```
pip install -r requirements.txt
```

## How to Use

### Playing the Game

Basic gameplay:
```
python main.py
```

Start with a specific strategy:
```
python main.py --strategy=train_lstm
```

### Training

Train on existing game logs before playing:
```
python main.py --offline-train --epochs=20
```

Train and then play with the trained model:
```
python main.py --offline-train --strategy=lstm
```

### Analysis

Analyze the game log without playing:
```
python main.py --analyze
```

## Game Controls

- **Left Arrow**: Play Rock
- **Down Arrow**: Play Paper
- **Right Arrow**: Play Scissors
- **Switch Strategy Button**: Toggle between AI strategies
- **Training Toggle Button**: Enable/disable LSTM training
- **Save Model Button**: Save the current LSTM model weights

## How the AI Learning Works

The LSTM (Long Short-Term Memory) neural network is trained to predict your next move based on your previous moves. The learning process works as follows:

1. The model tracks your move history
2. It creates a sequence of your last N moves
3. It tries to predict your next move based on this sequence
4. After you make your move, it compares its prediction with your actual move
5. It updates its internal weights to improve future predictions
6. The AI chooses the move that would beat its prediction of your next move

The more you play, the better the model gets at recognizing your patterns. Try to be unpredictable to beat the AI!

## Advanced Training Features

- **Experience Replay**: Stores previous moves and randomly samples from them during training
- **Data Augmentation**: Applies small variations to training data to improve generalization
- **Early Stopping**: Prevents overfitting by monitoring validation performance
- **Learning Rate Decay**: Reduces learning rate when improvement plateaus
- **Model Checkpointing**: Saves the best model based on validation accuracy

## Analysis Tools

The log analyzer provides insights into:
- Player move distribution
- Move transition probabilities
- AI prediction accuracy over time
- Win rates by strategy
- Pattern detection

Run the analyzer to visualize these metrics:
```
python main.py --analyze
```

## Extending the Project

### Adding New AI Strategies

1. Create a new strategy class in a separate file
2. Implement the `predict_next_move` method
3. Add the strategy to the `RPSModel` class in `rps_mvc.py`
4. Update the strategy switching code in the controller

### Training on External Data

You can train the model on external RPS data by formatting it as a CSV with the same structure as `rps_game_log.csv` and then running:
```
python main.py --offline-train --epochs=30
```

## License

This project is open source and available under the MIT License.