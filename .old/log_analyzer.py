import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, deque
import seaborn as sns
import os
from datetime import datetime

def load_game_log(file_path="rps_game_log.csv"):
    """Load the game log CSV file into a pandas DataFrame"""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None
        
    try:
        df = pd.read_csv(file_path)
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        print(f"Error loading game log: {e}")
        return None

def analyze_player_patterns(df, window_size=10):
    """Analyze patterns in player move selection"""
    if df is None or "player_move" not in df.columns:
        return None
        
    # Get the sequence of player moves
    moves = df["player_move"].tolist()
    
    # Count overall move distribution
    move_counts = Counter(moves)
    total = sum(move_counts.values())
    
    # Count transition probabilities (what move follows what)
    transitions = {}
    for i in range(len(moves) - 1):
        current_move = moves[i]
        next_move = moves[i + 1]
        
        if current_move not in transitions:
            transitions[current_move] = Counter()
        transitions[current_move][next_move] += 1
    
    # Calculate transition probabilities
    transition_probs = {}
    for current_move, next_moves in transitions.items():
        total_transitions = sum(next_moves.values())
        transition_probs[current_move] = {next_move: count / total_transitions 
                                        for next_move, count in next_moves.items()}
    
    # Look for higher-order patterns (e.g., after rock-paper, what comes next?)
    pattern_length = 2  # Example: looking at pairs of moves
    higher_order = {}
    
    for i in range(len(moves) - pattern_length):
        pattern = tuple(moves[i:i+pattern_length])
        next_move = moves[i+pattern_length]
        
        if pattern not in higher_order:
            higher_order[pattern] = Counter()
        higher_order[pattern][next_move] += 1
    
    # Calculate rolling frequencies (e.g., "is the player using rock more often lately?")
    window = deque(maxlen=window_size)
    rolling_frequencies = []
    
    for move in moves:
        window.append(move)
        window_counter = Counter(window)
        freqs = {move: count / len(window) for move, count in window_counter.items()}
        rolling_frequencies.append(freqs)
    
    # Return all analysis results
    return {
        "move_counts": move_counts,
        "transition_probs": transition_probs,
        "higher_order_patterns": higher_order,
        "rolling_frequencies": rolling_frequencies
    }

def analyze_prediction_accuracy(df):
    """Analyze the accuracy of AI predictions"""
    if df is None or "player_move" not in df.columns or "prediction" not in df.columns:
        return None
    
    # Filter out rows where prediction is None or NaN
    df_pred = df.dropna(subset=["prediction"])
    if df_pred.empty:
        return None
    
    # Shift player_move to get the next move
    df_pred["next_move"] = df_pred["player_move"].shift(-1)
    df_pred = df_pred.dropna(subset=["next_move"])
    
    # Calculate accuracy
    df_pred["correct"] = df_pred["prediction"] == df_pred["next_move"]
    accuracy = df_pred["correct"].mean()
    
    # Calculate per-move accuracy
    per_move_accuracy = {}
    for move in ["rock", "paper", "scissors"]:
        move_df = df_pred[df_pred["next_move"] == move]
        if not move_df.empty:
            per_move_accuracy[move] = move_df["correct"].mean()
    
    # Calculate rolling accuracy
    window_size = min(50, len(df_pred) // 2)  # Adjust window size based on data
    if window_size > 0:
        df_pred["rolling_accuracy"] = df_pred["correct"].rolling(window=window_size).mean()
    
    return {
        "overall_accuracy": accuracy,
        "per_move_accuracy": per_move_accuracy,
        "df_with_accuracy": df_pred
    }

def analyze_win_rates(df):
    """Analyze win rates by strategy and over time"""
    if df is None or "winner" not in df.columns:
        return None
    
    # Overall win rates
    win_counts = Counter(df["winner"])
    total_games = len(df)
    win_rates = {winner: count / total_games for winner, count in win_counts.items()}
    
    # Win rates by strategy
    if "strategy" in df.columns:
        strategy_win_rates = {}
        for strategy in df["strategy"].unique():
            strategy_df = df[df["strategy"] == strategy]
            if not strategy_df.empty:
                strategy_counts = Counter(strategy_df["winner"])
                strategy_total = len(strategy_df)
                strategy_win_rates[strategy] = {
                    winner: count / strategy_total for winner, count in strategy_counts.items()
                }
    else:
        strategy_win_rates = None
    
    # Win rates over time (rolling window)
    window_size = min(50, total_games // 5)  # Adjust window size based on data
    if window_size > 0:
        df["player_win"] = df["winner"] == "Player Wins"
        df["ai_win"] = df["winner"] == "AI Wins"
        df["draw"] = df["winner"] == "Draw"
        
        df["rolling_player_win_rate"] = df["player_win"].rolling(window=window_size).mean()
        df["rolling_ai_win_rate"] = df["ai_win"].rolling(window=window_size).mean()
        df["rolling_draw_rate"] = df["draw"].rolling(window=window_size).mean()
    
    return {
        "overall_win_rates": win_rates,
        "strategy_win_rates": strategy_win_rates,
        "df_with_rolling_rates": df
    }

def plot_move_distribution(analysis_results):
    """Plot the distribution of player moves"""
    if "move_counts" not in analysis_results:
        return None
    
    move_counts = analysis_results["move_counts"]
    moves = list(move_counts.keys())
    counts = list(move_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(moves, counts, color=['red', 'blue', 'green'])
    plt.title('Distribution of Player Moves')
    plt.xlabel('Move')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = sum(counts)
    for i, count in enumerate(counts):
        percentage = 100 * count / total
        plt.text(i, count + 0.5, f"{percentage:.1f}%", ha='center')
    
    plt.tight_layout()
    return plt.gcf()

def plot_transition_matrix(analysis_results):
    """Plot the transition probabilities as a heatmap"""
    if "transition_probs" not in analysis_results:
        return None
    
    transitions = analysis_results["transition_probs"]
    moves = ["rock", "paper", "scissors"]
    
    # Create transition matrix
    matrix = np.zeros((len(moves), len(moves)))
    for i, current_move in enumerate(moves):
        if current_move in transitions:
            for j, next_move in enumerate(moves):
                if next_move in transitions[current_move]:
                    matrix[i, j] = transitions[current_move][next_move]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=moves, yticklabels=moves)
    plt.title('Move Transition Probabilities')
    plt.xlabel('Next Move')
    plt.ylabel('Current Move')
    plt.tight_layout()
    return plt.gcf()

def plot_prediction_accuracy(accuracy_results):
    """Plot prediction accuracy over time"""
    if accuracy_results is None or "df_with_accuracy" not in accuracy_results:
        return None
    
    df = accuracy_results["df_with_accuracy"]
    
    plt.figure(figsize=(12, 6))
    
    # Overall accuracy line
    if "rolling_accuracy" in df.columns:
        plt.plot(df.index, df["rolling_accuracy"], label="Rolling Accuracy", color="blue")
    
    # Plot overall accuracy as horizontal line
    overall_acc = accuracy_results["overall_accuracy"]
    plt.axhline(y=overall_acc, color="red", linestyle="--", 
                label=f"Overall Accuracy: {overall_acc:.2f}")
    
    # Per-move accuracy
    per_move = accuracy_results["per_move_accuracy"]
    for move, acc in per_move.items():
        plt.axhline(y=acc, color={"rock": "red", "paper": "blue", "scissors": "green"}[move], 
                    linestyle=":", alpha=0.6, 
                    label=f"{move.capitalize()} Accuracy: {acc:.2f}")
    
    plt.title('LSTM Prediction Accuracy Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_win_rates_over_time(win_rate_results):
    """Plot win rates over time"""
    if win_rate_results is None or "df_with_rolling_rates" not in win_rate_results:
        return None
    
    df = win_rate_results["df_with_rolling_rates"]
    
    plt.figure(figsize=(12, 6))
    
    # Plot rolling win rates
    if "rolling_player_win_rate" in df.columns:
        plt.plot(df.index, df["rolling_player_win_rate"], label="Player Win Rate", color="blue")
        plt.plot(df.index, df["rolling_ai_win_rate"], label="AI Win Rate", color="red")
        plt.plot(df.index, df["rolling_draw_rate"], label="Draw Rate", color="green", linestyle="--")
    
    # Overall rates as horizontal lines
    overall_rates = win_rate_results["overall_win_rates"]
    for winner, rate in overall_rates.items():
        if winner == "Player Wins":
            color, style = "blue", "-."
        elif winner == "AI Wins":
            color, style = "red", "-."
        else:  # Draw
            color, style = "green", ":"
            
        plt.axhline(y=rate, color=color, linestyle=style, alpha=0.5,
                    label=f"Overall {winner}: {rate:.2f}")
    
    plt.title('Win Rates Over Time')
    plt.xlabel('Game Number')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def run_analysis(csv_file="rps_game_log.csv"):
    """Run all analyses and create visualizations"""
    # Load data
    df = load_game_log(csv_file)
    if df is None:
        print("No data to analyze")
        return None
    
    print(f"Loaded {len(df)} games from {csv_file}")
    
    # Run analyses
    pattern_analysis = analyze_player_patterns(df)
    prediction_accuracy = analyze_prediction_accuracy(df)
    win_rate_analysis = analyze_win_rates(df)
    
    # Create visualizations
    plots = {}
    
    if pattern_analysis:
        plots["move_distribution"] = plot_move_distribution(pattern_analysis)
        plots["transition_matrix"] = plot_transition_matrix(pattern_analysis)
    
    if prediction_accuracy:
        plots["prediction_accuracy"] = plot_prediction_accuracy(prediction_accuracy)
        
    if win_rate_analysis:
        plots["win_rates"] = plot_win_rates_over_time(win_rate_analysis)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    if pattern_analysis and "move_counts" in pattern_analysis:
        print("\nPlayer Move Distribution:")
        move_counts = pattern_analysis["move_counts"]
        total = sum(move_counts.values())
        for move, count in move_counts.items():
            print(f"  {move}: {count} ({count/total:.1%})")
    
    if prediction_accuracy:
        print("\nPrediction Accuracy:")
        print(f"  Overall: {prediction_accuracy['overall_accuracy']:.2%}")
        for move, acc in prediction_accuracy["per_move_accuracy"].items():
            print(f"  {move}: {acc:.2%}")
    
    if win_rate_analysis:
        print("\nOverall Win Rates:")
        for winner, rate in win_rate_analysis["overall_win_rates"].items():
            print(f"  {winner}: {rate:.2%}")
        
        if win_rate_analysis["strategy_win_rates"]:
            print("\nWin Rates by Strategy:")
            for strategy, rates in win_rate_analysis["strategy_win_rates"].items():
                print(f"  {strategy}:")
                for winner, rate in rates.items():
                    print(f"    {winner}: {rate:.2%}")
    
    # Show plots
    for name, plot in plots.items():
        if plot:
            plt.figure(plot.number)
            plt.show()
    
    return {
        "dataframe": df,
        "pattern_analysis": pattern_analysis,
        "prediction_accuracy": prediction_accuracy,
        "win_rate_analysis": win_rate_analysis,
        "plots": plots
    }

if __name__ == "__main__":
    run_analysis()