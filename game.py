class Game:
    def __init__(self):
        self.model = None
        self.view = None
        self.controller = None

        self.rules = None

        self.player1 = None
        self.player2 = None

        self.frame = None

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model
    
    def set_view(self, view):
        self.view = view

    def get_view(self):
        return self.view
    
    def set_controller(self, controller):
        self.controller = controller

    def get_controller(self):
        return self.controller

    def set_rules(self, rules):
        self.rules = rules

    def get_rules(self):
        return self.rules
    
    def set_player1(self, player1):
        self.player1 = player1

    def get_player1(self):
        return self.player1
    
    def set_player2(self, player2):
        self.player2 = player2

    def get_player2(self):
        return self.player2
    
    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame

class GameRules:
    def __init__(self):
        self.game = None
        self.rules = None
        self.best_of = None

    def set_game(self, game):
        self.game = game
    
    def get_game(self):
        return self.game

    def set_rules(self):
        self.rules = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper"
        }

    def get_rules(self):
        return self.rules
    
    def set_best_of(self, best_of):
        self.best_of = best_of

    def get_best_of(self):
        return self.best_of

class GameFrame:
    def __init__(self):
        self.game = None
        
        self.player1 = None
        self.player2 = None

        self.winner = None
    
    def set_game(self, game):
        self.game = game

    def get_game(self):
        return self.game

    def set_player1(self, player1):
        self.player1 = player1
    
    def get_player1(self):
        return self.player1

    def set_player2(self, player2):
        self.player2 = player2

    def get_player2(self):
        return self.player2
    
    # self.winner is None by default.
    # If it's a draw, self.winner will still be None.
    def set_winner(self, winner=None):
        if winner is None:
            if self.player1.get_choice() == self.player2.get_choice():
                self.winner = None
            if self.player1.get_choice() == self.game.rules.get_rules()[self.player2.get_choice()]:
                self.winner = self.player1
            else:
                self.winner = self.player2
        else:
            self.winner = winner

    def get_winner(self):
        return self.winner