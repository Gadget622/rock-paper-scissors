class RPSController:
    def __init__(self):
        self.model = None
        self.view = None
    
    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model

    def set_view(self, view):
        self.view = view

    def get_view(self):
        return self.view

    def play_game(self):
        while True:
            user_choice = self.view.get_user_choice()
            if user_choice is None:
                break
            self.model.get_computer_choice()
            result = self.model.determine_winner()
            self.view.display_result(result)