class Player:
    def __init__(self):
        self.name = None
        self.game = None
        self.state = None
        self.choice = None
    
    def set_name(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def set_game(self, game):
        self.game = game

    def get_game(self):
        return self.game

    # In regular rock-paper-scissors, there is only one unique state with three options: rock, paper, and scissors.
    # In a more complex game, there could be multiple states with different options.
    # For example, in a game with multiple players or different rounds, the options might change based on the current state of the game.
    def set_state(self, state_name):
        
        print(f"Setting state for {self.name} with state name {state_name}")
        self.state = PlayerState(name=state_name)
    
    def get_state(self):
        return self.state
    
    def set_options(self, options = None):
        if options is None:
            self.options = self.state.get_options()
        else:
            self.options = options
    
    def get_options(self):
        return self.options

    # Choices will be made once the game starts
    def set_choice(self, choice=None):
        if choice is None:
            self.choice = input()
        else:
            self.choice = choice

    def get_choice(self):
        return self.choice

class AIPlayer(Player):
    def __init__(self):
        super().__init__()
        self.ai_strategy = None

    def set_ai_strategy(self, ai_strategy):
        self.ai_strategy = ai_strategy

    def get_ai_strategy(self):
        return self.ai_strategy
    
    def set_choice(self, choice=None):
        if choice is None:
            if self.ai_strategy == 'random':
                import random
                options = self.get_options()
                self.choice = random.choice(options)
        else:
            self.choice = choice

    def get_choice(self):
        return super().get_choice()
    
# States are explicitly defined by the options available to the players.
class PlayerState:
    def __init__(self, name=None):
        self.name = self.set_name(name)
        self.game = None
        self.frame = None
        self.options = None


    def set_name(self, name=None):
        if name == 'rps':
            self.set_options(
                {
                    "rock",
                    "paper",
                    "scissors"
                }
            )
    
    def get_name(self):
        return self.name

    def set_game(self, game):
        self.game = game

    def get_game(self):
        return self.game
    
    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame
    
    def set_options(self, options):
        self.options = options

    def get_options(self):
        return self.options
    
