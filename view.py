class RPSView:
    def __init__(self):
        self.model = None
        self.controller = None
    
    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model
    
    def set_controller(self, controller):
        self.controller = controller

    def get_controller(self):
        return self.controller

    def display_welcome(self):
        print("Welcome to Rock, Paper, Scissors!")
        print("Choose your weapon:")
        print("1. Rock")
        print("2. Paper")
        print("3. Scissors")

    def get_user_choice(self):
        choice = input("Enter your choice (1-3): ")
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Invalid choice. Please try again.")
            return self.get_user_choice()

    def display_result(self, result):
        if result == "draw":
            print("It's a draw!")
        elif result == "win":
            print("You win!")
        else:
            print("You lose!")