class RPSModel:
    def __init__(self):
        self.view = None
        self.controller = None

    def set_view(self, view):
        self.view = view

    def get_view(self):
        return self.view
    
    def set_controller(self, controller):
        self.controller = controller
    
    def get_controller(self):
        return self.controller