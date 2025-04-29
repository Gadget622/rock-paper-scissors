from model import RPSModel
from view import RPSView
from controller import RPSController

from game import Game, GameFrame
from player import Player, AIPlayer

def main():
    # Initialize the model, view, and controller
    model = RPSModel()
    view = RPSView()
    controller = RPSController()

    model.set_view(view)
    model.set_controller(controller)

    view.set_model(model)
    view.set_controller(controller)

    controller.set_model(model)
    controller.set_view(view)

    # Initializing the rules
    rules = {
        'rock' : ('scissors','surfer','donut'),
        'paper' : ('unicorn','rock','fire'),    # The last one doesn't make sense to me either, but that's what the rules say
        'scissors' : ('donut','paper','surfer'),
        'surfer' : ('donut','paper','unicorn'),
        'unicorn' : ('scissors','rock','fire'),
        'donut' : ('fire','unicorn','paper')
    }
    rps_options = list(rules.keys())

    
    # Initializing players

    player1 = Player()
    player1.set_name('Player 1')
    player1.set_options(options=rps_options)
    print(f'{player1.name} has the following options: {player1.options}')
    
    player2 = AIPlayer()
    player2.set_name('Player 2')
    player2.set_options(options=rps_options)
    print(f'{player2.name} has the following options: {player2.options}')
    player2.set_ai_strategy('random')
    
    print('Here are the rules!')
    for option in rules:
        print(f'{option} beats {rules[option]}')

    # Initialize the game
    game = Game()
    game.set_model(model)
    game.set_view(view)
    game.set_controller(controller)
    game.set_player1(player1)
    game.set_player2(player2)
    game.set_rules(rules)
    
    player1.set_game(game)
    player2.set_game(game)
    
    playing = True
    max_wins = 3
    player1_wins = 0
    player2_wins = 0
    set_winner = None

    while playing:
        # Initialize the frame
        frame = GameFrame()
        frame.set_game(game)
        frame.set_player1(player1)
        frame.set_player2(player2)
        game.set_frame(frame)
        
        
        print(f'Time to make a choice! {list(rules.keys())}')
        player1.set_choice()
        print(f'{player1.name} chose {player1.get_choice()}')
        player2.set_choice()
        print(f'{player2.name} chose {player2.get_choice()}')

        if player1.get_choice() in rules[player2.get_choice()]:
            frame.set_winner(winner = player2)
            player2_wins += 1
            print(f'{player2.name} wins that round!')
        elif player2.get_choice() in rules[player1.get_choice()]:
            frame.set_winner(winner = player1)
            player1_wins += 1
            print(f'{player1.name} wins that round!')

        # Checking if someone won the set
        if player1_wins >= max_wins:
            playing = False
            set_winner = 'Player 1'

        elif player2_wins >= max_wins:
            playing = False
            set_winner = 'Player 2'
    
    print(f'The winner is {set_winner} in the best of {2*max_wins-1}!')


if __name__ == '__main__':
    main()