import pygame
import random
import csv
import os
import sys
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Try to import the transformer model
try:
    from transformer_rps_model import TransformerStrategy
    print("Transformer model available")
except ImportError:
    print("Transformer model not available")

# Try to import the enhanced LSTM model
try:
    from enhanced_lstm_model import LSTMStrategy
except ImportError:
    # Fall back to original LSTM model if enhanced version is not available
    try:
        from enhanced_lstm_model import LSTMStrategy
        print("Using original LSTM model instead of enhanced version")
    except ImportError:
        print("Warning: Could not import any LSTM model")
        class LSTMStrategy:
            def __init__(self):
                self.last_prediction = None
            def predict_next_move(self, moves):
                self.last_prediction = random.choice(["rock", "paper", "scissors"])
                return self.last_prediction
            def counter_move(self, move):
                if move == "rock": return "paper"
                elif move == "paper": return "scissors"
                else: return "rock"

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600  # Increased height for metrics display
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 0, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
PURPLE = (180, 0, 180)
FONT = pygame.font.SysFont("arial", 24)
SMALL_FONT = pygame.font.SysFont("arial", 16)
MOVES = ["rock", "paper", "scissors"]
KEY_BINDINGS = {
    pygame.K_LEFT: "rock",
    pygame.K_DOWN: "paper",
    pygame.K_RIGHT: "scissors"
}
CSV_FILE = "rps_game_log.csv"
TRAINING_INTERVAL = 5  # Train every N moves

# Determine current game number
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["game_number", "timestamp", "player_move", "ai_move", "winner", "strategy", "prediction", "pattern_detected"])
    game_number = 1
else:
    with open(CSV_FILE, mode='r') as file:
        lines = file.readlines()
        if len(lines) <= 1:
            game_number = 1
        else:
            try:
                last_game_number = int(lines[-1].split(',')[0])
                game_number = last_game_number + 1
            except (ValueError, IndexError):
                game_number = 1

# Model
class RPSModel:
    def __init__(self, strategy="random", seed=None):
        self.player_move = None
        self.ai_move = None
        self.strategy = strategy
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.total_games = 0
        self.prev_player_moves = []
        self.training_enabled = True
        self.auto_save_interval = 50  # Save model every 50 games
        self.training_metrics = {"losses": [], "accuracies": [], "games": []}
        self.pattern_detected = None
        
        if seed is not None:
            random.seed(seed)

        # Initialize models
        self.lstm_model = LSTMStrategy() if strategy in ["lstm", "train_lstm"] else None
        
        # Initialize transformer model if needed
        if strategy in ["transformer", "train_transformer"]:
            try:
                self.transformer_model = TransformerStrategy()
            except Exception as e:
                print(f"Error initializing transformer model: {e}")
                self.transformer_model = None
        else:
            self.transformer_model = None
        
        # Load historic data for initial training if available
        self.load_historic_data()

    def load_historic_data(self):
        """Load historic game data for initial training"""
        if not os.path.exists(CSV_FILE):
            return
            
        try:
            with open(CSV_FILE, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                historic_moves = []
                for row in reader:
                    if len(row) >= 3:  # Make sure we have player_move
                        historic_moves.append(row[2])  # player_move column
                
                # Initial training on LSTM model
                if len(historic_moves) > 10 and self.lstm_model and hasattr(self.lstm_model, 'train_on_move'):
                    print(f"Training LSTM on {len(historic_moves)} historic moves")
                    for i in range(len(historic_moves) - 1):
                        if i % 10 == 0:  # Train every 10th move (for speed)
                            history = historic_moves[:i+1]
                            actual = historic_moves[i+1]
                            self.lstm_model.train_on_move(history, actual)
                            
                # Initial training on Transformer model
                if len(historic_moves) > 10 and self.transformer_model and hasattr(self.transformer_model, 'train_on_move'):
                    print(f"Training Transformer on {len(historic_moves)} historic moves")
                    for i in range(len(historic_moves) - 1):
                        if i % 10 == 0:  # Train every 10th move (for speed)
                            history = historic_moves[:i+1]
                            actual = historic_moves[i+1]
                            self.transformer_model.train_on_move(history, actual)
        except Exception as e:
            print(f"Error loading historic data: {e}")

    def reset_stats(self):
        self.player_wins = 0
        self.ai_wins = 0
        self.draws = 0
        self.total_games = 0
        self.pattern_detected = None

    def set_strategy(self, new_strategy):
        old_strategy = self.strategy
        self.strategy = new_strategy
        
        # Only reset stats if changing fundamental strategy type
        strategy_types = {
            "random": "random",
            "lstm": "lstm", 
            "train_lstm": "lstm",
            "transformer": "transformer",
            "train_transformer": "transformer"
        }
        
        if strategy_types.get(old_strategy, "") != strategy_types.get(new_strategy, ""):
            self.reset_stats()
            
        # Initialize Transformer if needed
        if new_strategy in ["transformer", "train_transformer"] and not self.transformer_model:
            try:
                self.transformer_model = TransformerStrategy()
            except Exception as e:
                print(f"Error initializing transformer model: {e}")
                self.transformer_model = None
                
        # Initialize LSTM if needed
        if new_strategy in ["lstm", "train_lstm"] and not self.lstm_model:
            try:
                self.lstm_model = LSTMStrategy()
            except Exception as e:
                print(f"Error initializing LSTM model: {e}")
                self.lstm_model = None

    def toggle_training(self):
        self.training_enabled = not self.training_enabled
        return self.training_enabled

    def save_current_model(self):
        """Save the current model based on strategy"""
        if self.strategy in ["lstm", "train_lstm"] and self.lstm_model and hasattr(self.lstm_model, 'save_model'):
            return self.lstm_model.save_model()
        elif self.strategy in ["transformer", "train_transformer"] and self.transformer_model and hasattr(self.transformer_model, 'save_model'):
            return self.transformer_model.save_model()
        return False

    def get_ai_move(self):
        """Get AI move based on current strategy"""
        if self.strategy == "random":
            return random.choice(MOVES)
        elif self.strategy in ["lstm", "train_lstm"] and self.lstm_model and self.prev_player_moves:
            predicted_move = self.lstm_model.predict_next_move(self.prev_player_moves)
            return self.lstm_model.counter_move(predicted_move)
        elif self.strategy in ["transformer", "train_transformer"] and self.transformer_model and self.prev_player_moves:
            # First check for explicit patterns
            self.pattern_detected = None
            if hasattr(self.transformer_model, 'pattern_detector'):
                pattern_pred, confidence = self.transformer_model.pattern_detector.detect_pattern(self.prev_player_moves)
                if pattern_pred and confidence > 0.7:
                    self.pattern_detected = f"{pattern_pred} ({confidence:.2f})"
            
            return self.transformer_model.select_move(self.prev_player_moves)
        return random.choice(MOVES)

    def play_round(self, player_move):
        """Play a round with the current move"""
        self.player_move = player_move
        
        # Capture prediction before adding current move
        predicted_next = None
        if self.strategy in ["lstm", "train_lstm"] and self.lstm_model and self.prev_player_moves:
            predicted_next = self.lstm_model.predict_next_move(self.prev_player_moves)
        elif self.strategy in ["transformer", "train_transformer"] and self.transformer_model and self.prev_player_moves:
            predicted_next = self.transformer_model.predict_next_move(self.prev_player_moves)
        
        # Get AI move
        self.ai_move = self.get_ai_move()
        winner = self.get_winner()

        # Update statistics
        self.total_games += 1
        if winner == "Player Wins":
            self.player_wins += 1
        elif winner == "AI Wins":
            self.ai_wins += 1
        else:
            self.draws += 1

        # Add move to history
        self.prev_player_moves.append(player_move)
        
        # Train model if enabled
        loss = None
        if self.training_enabled:
            if self.strategy == "train_lstm" and self.lstm_model and len(self.prev_player_moves) > 1:
                if hasattr(self.lstm_model, 'train_on_move'):
                    # Train on the actual move that was just played
                    previous_moves = self.prev_player_moves[:-1]  # All but the last move
                    actual_move = self.prev_player_moves[-1]     # The move that was just played
                    
                    try:
                        loss = self.lstm_model.train_on_move(previous_moves, actual_move)
                        
                        # Update metrics
                        if loss is not None and hasattr(self.lstm_model, 'get_training_metrics'):
                            metrics = self.lstm_model.get_training_metrics()
                            self.training_metrics["losses"].append(metrics.get("avg_loss", 0))
                            self.training_metrics["accuracies"].append(metrics.get("avg_accuracy", 0))
                            self.training_metrics["games"].append(self.total_games)
                    except Exception as e:
                        print(f"Error during LSTM training: {e}")
            
            elif self.strategy == "train_transformer" and self.transformer_model and len(self.prev_player_moves) > 1:
                if hasattr(self.transformer_model, 'train_on_move'):
                    # Train on the actual move that was just played
                    previous_moves = self.prev_player_moves[:-1]  # All but the last move
                    actual_move = self.prev_player_moves[-1]     # The move that was just played
                    
                    try:
                        loss = self.transformer_model.train_on_move(previous_moves, actual_move)
                        
                        # Update metrics
                        if loss is not None and hasattr(self.transformer_model, 'get_training_metrics'):
                            metrics = self.transformer_model.get_training_metrics()
                            self.training_metrics["losses"].append(metrics.get("avg_loss", 0))
                            self.training_metrics["accuracies"].append(metrics.get("avg_accuracy", 0))
                            self.training_metrics["games"].append(self.total_games)
                    except Exception as e:
                        print(f"Error during Transformer training: {e}")
            
            # Periodic save
            if self.total_games % self.auto_save_interval == 0:
                self.save_current_model()

        # Log the round to CSV
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            prediction_str = predicted_next if predicted_next else "None"
            pattern_str = self.pattern_detected if self.pattern_detected else "None"
            writer.writerow([
                game_number,
                datetime.now().isoformat(),
                self.player_move,
                self.ai_move,
                winner,
                self.strategy,
                prediction_str,
                pattern_str
            ])
            
        return loss

    def get_winner(self):
        if self.player_move == self.ai_move:
            return "Draw"
        elif (self.player_move == "rock" and self.ai_move == "scissors") or \
             (self.player_move == "scissors" and self.ai_move == "paper") or \
             (self.player_move == "paper" and self.ai_move == "rock"):
            return "Player Wins"
        else:
            return "AI Wins"
    
    def get_training_metrics(self):
        """Get training metrics from current model"""
        if self.strategy in ["lstm", "train_lstm"] and self.lstm_model and hasattr(self.lstm_model, 'get_training_metrics'):
            return self.lstm_model.get_training_metrics()
        elif self.strategy in ["transformer", "train_transformer"] and self.transformer_model and hasattr(self.transformer_model, 'get_training_metrics'):
            return self.transformer_model.get_training_metrics()
        return None

# View
class RPSView:
    def __init__(self, screen):
        self.screen = screen
        self.strategy_button = pygame.Rect(WIDTH//2 - 100, HEIGHT - 140, 200, 40)
        self.training_button = pygame.Rect(WIDTH//2 - 100, HEIGHT - 90, 200, 40)
        self.save_button = pygame.Rect(WIDTH//2 - 100, HEIGHT - 40, 200, 40)
        
        # For metrics graph
        self.metrics_surface = None
        self.metrics_rect = pygame.Rect(50, 250, WIDTH - 100, 180)
        self.last_update = 0
        self.update_interval = 10  # Update graph every 10 games

    def create_metrics_surface(self, model):
        """Create a surface for displaying training metrics"""
        if len(model.training_metrics["games"]) < 2:
            return None
            
        try:
            # Create matplotlib figure
            fig, ax1 = plt.subplots(figsize=(7, 1.6))
            
            # Plot loss
            losses = model.training_metrics["losses"]
            games = model.training_metrics["games"]
            if losses and games:
                ax1.plot(games, losses, 'b-', label='Loss')
                ax1.set_xlabel('Games')
                ax1.set_ylabel('Loss', color='b')
                ax1.tick_params(axis='y', labelcolor='b')
                
                # Plot accuracy on secondary y-axis
                ax2 = ax1.twinx()
                accuracy = model.training_metrics["accuracies"]
                if accuracy:
                    ax2.plot(games, accuracy, 'r-', label='Accuracy')
                    ax2.set_ylabel('Accuracy', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.set_ylim(0, 1)
                
                plt.title('Training Metrics')
                plt.tight_layout()
                
                # Convert to pygame surface
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                renderer = canvas.get_renderer()
                raw_data = renderer.tostring_rgb()
                size = canvas.get_width_height()
                
                surf = pygame.image.fromstring(raw_data, size, "RGB")
                plt.close(fig)
                return surf
        except Exception as e:
            print(f"Error creating metrics surface: {e}")
        return None

    def draw(self, model):
        self.screen.fill(WHITE)

        # Draw move history and results
        if model.player_move and model.ai_move:
            player_text = FONT.render(f"Player: {model.player_move}", True, BLACK)
            ai_text = FONT.render(f"AI: {model.ai_move}", True, BLACK)
            result_text = FONT.render(model.get_winner(), True, BLACK)
            
            self.screen.blit(player_text, (50, 50))
            self.screen.blit(ai_text, (50, 90))
            self.screen.blit(result_text, (50, 130))
            
            # Show prediction
            if model.strategy in ["lstm", "train_lstm"] and model.lstm_model and hasattr(model.lstm_model, 'last_prediction'):
                predicted = model.lstm_model.last_prediction
                if predicted:
                    pred_text = FONT.render(f"AI predicted you would play: {predicted}", True, BLUE)
                    self.screen.blit(pred_text, (50, 170))
            elif model.strategy in ["transformer", "train_transformer"] and model.transformer_model and hasattr(model.transformer_model, 'last_prediction'):
                predicted = model.transformer_model.last_prediction
                if predicted:
                    pred_text = FONT.render(f"AI predicted you would play: {predicted}", True, PURPLE)
                    self.screen.blit(pred_text, (50, 170))
                    
                    # Show if pattern was detected
                    if model.pattern_detected:
                        pattern_text = FONT.render(f"Pattern detected: {model.pattern_detected}", True, PURPLE)
                        self.screen.blit(pattern_text, (50, 200))
        else:
            instructions = FONT.render("Press ← for Rock, ↓ for Paper, → for Scissors", True, BLACK)
            self.screen.blit(instructions, (50, 100))

        # Draw game statistics
        if model.total_games > 0:
            player_pct = model.player_wins / model.total_games * 100
            ai_pct = model.ai_wins / model.total_games * 100
            draw_pct = model.draws / model.total_games * 100
            
            ai_ratio = model.ai_wins / model.player_wins if model.player_wins > 0 else float('inf')
            player_ratio = model.player_wins / model.ai_wins if model.ai_wins > 0 else float('inf')
            
            # Format ratios with maximum of 2 decimal places
            ai_ratio_str = f"{ai_ratio:.2f}" if ai_ratio != float('inf') else "∞" 
            player_ratio_str = f"{player_ratio:.2f}" if player_ratio != float('inf') else "∞"
        else:
            player_pct = ai_pct = draw_pct = 0
            ai_ratio_str = player_ratio_str = "0.00"

        stats = [
            f"Strategy: {model.strategy}",
            f"Total: {model.total_games}",
            f"Player Wins: {model.player_wins} ({player_pct:.1f}%)",
            f"AI Wins: {model.ai_wins} ({ai_pct:.1f}%)",
            f"Draws: {model.draws} ({draw_pct:.1f}%)",
            f"Player Ratio: {player_ratio_str}",
            f"AI Ratio: {ai_ratio_str}"
        ]
        
        # Add training metrics if available
        if model.strategy in ["train_lstm", "train_transformer"]:
            metrics = model.get_training_metrics()
            if metrics:
                stats.append("")
                stats.append(f"Training: {'Enabled' if model.training_enabled else 'Disabled'}")
                stats.append(f"Samples: {metrics.get('total_samples', 0)}")
                stats.append(f"Avg Loss: {metrics.get('avg_loss', 0):.4f}")
                stats.append(f"Prediction Accuracy: {metrics.get('avg_accuracy', 0):.2f}")
                
                # Add transformer-specific metrics
                if model.strategy == "train_transformer" and "transformer_weight" in metrics:
                    stats.append(f"Transformer weight: {metrics.get('transformer_weight', 0):.2f}")
                    stats.append(f"Pattern detector weight: {metrics.get('pattern_detector_weight', 0):.2f}")
        
        for i, text in enumerate(stats):
            stat_text = FONT.render(text, True, BLACK)
            self.screen.blit(stat_text, (500, 30 + i * 30))

        # Draw metrics graph if needed
        if model.strategy in ["train_lstm", "train_transformer"] and len(model.training_metrics["games"]) > 0:
            # Only update graph periodically to save performance
            if self.metrics_surface is None or \
               (model.total_games - self.last_update >= self.update_interval):
                self.metrics_surface = self.create_metrics_surface(model)
                self.last_update = model.total_games
                
            if self.metrics_surface:
                # Draw background for graph
                pygame.draw.rect(self.screen, GRAY, self.metrics_rect)
                
                # Draw graph
                self.screen.blit(self.metrics_surface, self.metrics_rect)
                
                # Draw border
                pygame.draw.rect(self.screen, BLACK, self.metrics_rect, 2)

        # Draw UI buttons
        pygame.draw.rect(self.screen, BLACK, self.strategy_button, 2)
        strategy_text = FONT.render("Switch Strategy", True, BLACK)
        self.screen.blit(strategy_text, (self.strategy_button.x + 20, self.strategy_button.y + 5))
        
        # Only show training/save buttons if using a trainable model
        if model.strategy in ["train_lstm", "train_transformer"]:
            # Training toggle button
            color = GREEN if model.training_enabled else RED
            pygame.draw.rect(self.screen, color, self.training_button)
            pygame.draw.rect(self.screen, BLACK, self.training_button, 2)
            training_text = FONT.render(f"Training: {'ON' if model.training_enabled else 'OFF'}", True, BLACK)
            self.screen.blit(training_text, (self.training_button.x + 30, self.training_button.y + 5))
            
            # Save model button
            pygame.draw.rect(self.screen, BLUE, self.save_button)
            pygame.draw.rect(self.screen, BLACK, self.save_button, 2)
            save_text = FONT.render("Save Model", True, WHITE)
            self.screen.blit(save_text, (self.save_button.x + 40, self.save_button.y + 5))

        pygame.display.flip()

# Controller
class RPSController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.message = None
        self.message_timer = 0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in KEY_BINDINGS:
            move = KEY_BINDINGS[event.key]
            self.model.play_round(move)
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Strategy button clicked - cycle through strategies
            if self.view.strategy_button.collidepoint(event.pos):
                if self.model.strategy == "random":
                    self.model.set_strategy("train_lstm")
                    self.show_message("Switched to LSTM with Training")
                elif self.model.strategy == "train_lstm":
                    self.model.set_strategy("lstm")
                    self.show_message("Switched to LSTM (no training)")
                elif self.model.strategy == "lstm":
                    self.model.set_strategy("train_transformer")
                    self.show_message("Switched to Transformer with Training")
                elif self.model.strategy == "train_transformer":
                    self.model.set_strategy("transformer")
                    self.show_message("Switched to Transformer (no training)")
                else:
                    self.model.set_strategy("random")
                    self.show_message("Switched to Random Strategy")
            
            # Training toggle button
            elif self.view.training_button.collidepoint(event.pos) and self.model.strategy in ["train_lstm", "train_transformer"]:
                training_enabled = self.model.toggle_training()
                status = "Enabled" if training_enabled else "Disabled"
                self.show_message(f"Training {status}")
            
            # Save model button
            elif self.view.save_button.collidepoint(event.pos) and self.model.strategy in ["train_lstm", "lstm", "train_transformer", "transformer"]:
                if self.model.save_current_model():
                    self.show_message("Model Saved!")
                else:
                    self.show_message("Failed to save model")
    
    def update(self):
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= 1
            if self.message_timer == 0:
                self.message = None
    
    def show_message(self, text, duration=90):  # 90 frames = ~1.5 seconds at 60fps
        self.message = text
        self.message_timer = duration
    
    def draw_message(self, screen):
        if self.message:
            # Draw message box
            msg_font = pygame.font.SysFont("arial", 20, bold=True)
            text_surface = msg_font.render(self.message, True, BLACK)
            
            padding = 10
            box_width = text_surface.get_width() + padding * 2
            box_height = text_surface.get_height() + padding * 2
            
            box_x = (WIDTH - box_width) // 2
            box_y = 10
            
            # Draw semi-transparent background
            s = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
            s.fill((255, 255, 200, 220))  # Light yellow with alpha
            screen.blit(s, (box_x, box_y))
            
            # Draw border
            pygame.draw.rect(screen, BLACK, (box_x, box_y, box_width, box_height), 2)
            
            # Draw text
            screen.blit(text_surface, (box_x + padding, box_y + padding))
                
# Main function
def main(initial_strategy=None):
    """Main game function, can accept an initial strategy"""
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RPS Game with AI Learning")

    # Get strategy from command line if not provided
    if initial_strategy is None:
        import argparse
        parser = argparse.ArgumentParser(description='RPS Game')
        parser.add_argument('--strategy', type=str, default='random', 
                           choices=['random', 'lstm', 'train_lstm', 'transformer', 'train_transformer'],
                           help='AI strategy to use')
        args, _ = parser.parse_known_args()
        initial_strategy = args.strategy

    model = RPSModel(strategy=initial_strategy, seed=None)
    view = RPSView(screen)
    controller = RPSController(model, view)

    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save model before exit
                model.save_current_model()
                running = False
            controller.handle_event(event)
        
        controller.update()
        view.draw(model)
        
        # Draw any active messages
        if controller.message:
            controller.draw_message(screen)
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()