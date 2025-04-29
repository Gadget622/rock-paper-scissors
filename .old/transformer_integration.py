import os
import sys

# Add transformer model to strategies
def add_transformer_to_rps_mvc():
    """Modify the RPS MVC script to incorporate the transformer model"""
    
    # First, make sure transformer model is available
    if not os.path.exists('transformer_rps_model.py'):
        print("Transformer model file not found!")
        print("Please create 'transformer_rps_model.py' with the provided code first.")
        return False
    
    # Check if rps_mvc.py exists
    if not os.path.exists('rps_mvc.py'):
        print("rps_mvc.py file not found!")
        return False
    
    # Backup existing file
    try:
        with open('rps_mvc.py', 'r') as f:
            original_content = f.read()
        
        with open('rps_mvc.py.bak', 'w') as f:
            f.write(original_content)
        print("Created backup of rps_mvc.py as rps_mvc.py.bak")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Add transformer import
    new_content = original_content
    
    # Add transformer import
    try:
        import_idx = new_content.find("# Try to import the enhanced LSTM model")
        if import_idx != -1:
            import_section = """# Try to import the transformer model
try:
    from transformer_rps_model import TransformerStrategy
    print("Transformer model available")
except ImportError:
    print("Transformer model not available")
    
# Try to import the enhanced LSTM model"""
            
            new_content = new_content[:import_idx] + import_section + new_content[import_idx:]
        
        # Add transformer as a strategy option
        strategy_switch_idx = new_content.find("elif self.model.strategy == \"train_lstm\":")
        if strategy_switch_idx != -1:
            strategy_line = """                elif self.model.strategy == "transformer":
                    self.model.set_strategy("lstm")
                    self.show_message("Switched to Transformer Model")
                """
            # Find the appropriate line after the elif
            next_line_idx = new_content.find("else:", strategy_switch_idx)
            if next_line_idx != -1:
                new_content = new_content[:next_line_idx] + strategy_line + new_content[next_line_idx:]
        
        # Update the strategy options in the RPSModel.set_strategy method
        set_strategy_idx = new_content.find("def set_strategy(self, new_strategy):")
        if set_strategy_idx != -1:
            # Find the end of the method
            lstm_init_idx = new_content.find("if new_strategy in [\"lstm\", \"train_lstm\"]", set_strategy_idx)
            if lstm_init_idx != -1:
                transformer_init = """        # Initialize Transformer if needed
        if new_strategy == "transformer" and not hasattr(self, 'transformer_model'):
            try:
                self.transformer_model = TransformerStrategy()
            except Exception as e:
                print(f"Error initializing transformer model: {e}")
                self.transformer_model = None
                
        """
                new_content = new_content[:lstm_init_idx] + transformer_init + new_content[lstm_init_idx:]
        
        # Update the get_ai_move method to use transformer
        get_ai_move_idx = new_content.find("def get_ai_move(self):")
        if get_ai_move_idx != -1:
            # Find the end of the method
            end_idx = new_content.find("return random.choice(MOVES)", get_ai_move_idx)
            if end_idx != -1:
                transformer_move = """        elif self.strategy == "transformer" and self.prev_player_moves and hasattr(self, 'transformer_model') and self.transformer_model:
            return self.transformer_model.select_move(self.prev_player_moves)
        """
                new_content = new_content[:end_idx] + transformer_move + new_content[end_idx:]
        
        # Write the modified content back to the file
        with open('rps_mvc.py', 'w') as f:
            f.write(new_content)
        
        print("Successfully added transformer model to rps_mvc.py")
        return True
        
    except Exception as e:
        print(f"Error modifying file: {e}")
        # Restore original file from backup
        try:
            with open('rps_mvc.py.bak', 'r') as f:
                original_content = f.read()
            
            with open('rps_mvc.py', 'w') as f:
                f.write(original_content)
            print("Restored original file from backup")
        except Exception as e2:
            print(f"Error restoring backup: {e2}")
        
        return False

def update_main_script():
    """Update the main script to include transformer model option"""
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("main.py file not found!")
        return False
    
    # Backup existing file
    try:
        with open('main.py', 'r') as f:
            original_content = f.read()
        
        with open('main.py.bak', 'w') as f:
            f.write(original_content)
        print("Created backup of main.py as main.py.bak")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Update strategy choices
    try:
        parser_idx = original_content.find("parser.add_argument('--strategy', type=str, default='random'")
        if parser_idx != -1:
            end_idx = original_content.find(')', parser_idx)
            if end_idx != -1:
                new_strategy_line = "parser.add_argument('--strategy', type=str, default='random', choices=['random', 'lstm', 'train_lstm', 'transformer']"
                new_content = original_content[:parser_idx] + new_strategy_line + original_content[end_idx:]
                
                # Write the modified content back to the file
                with open('main.py', 'w') as f:
                    f.write(new_content)
                
                print("Successfully updated main.py with transformer option")
                return True
        
        print("Could not find strategy argument in main.py")
        return False
        
    except Exception as e:
        print(f"Error modifying main.py: {e}")
        # Restore original file from backup
        try:
            with open('main.py.bak', 'r') as f:
                original_content = f.read()
            
            with open('main.py', 'w') as f:
                f.write(original_content)
            print("Restored original file from backup")
        except Exception as e2:
            print(f"Error restoring backup: {e2}")
        
        return False

if __name__ == "__main__":
    print("Adding transformer model to RPS game...")
    
    # First, create the transformer model file
    if not os.path.exists('transformer_rps_model.py'):
        print("Please create the transformer_rps_model.py file first with the provided code.")
        print("Then run this script again.")
        sys.exit(1)
    
    # Then update the MVC implementation
    success1 = add_transformer_to_rps_mvc()
    
    # Finally update the main script
    success2 = update_main_script()
    
    if success1 and success2:
        print("\nSuccessfully integrated transformer model!")
        print("\nYou can now run the game with the transformer model using:")
        print("python main.py --strategy=transformer")
    else:
        print("\nIntegration had some issues. Check the error messages above.")