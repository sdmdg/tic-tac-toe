import random
from utlis import show_msg


def find_ai_move(board, player_move, player_symbol=False, ai_symbol=True, app=None):
    if not (0 <= player_move < 9):
        print("Invalid player move.")
        return None

    # Make the player's move on a copy of the board
    new_board = board.copy()
    new_board[player_move] = player_symbol

    # Check for a winning move for the AI
    for i in range(9):
        if new_board[i] == None:
            new_board[i] = ai_symbol
            if check_winner(convert_to_matrix(new_board)) == ai_symbol:
                print("I Won.")
                show_msg(app=app, msg="I Won. Better luck next time :)")
                return i
            new_board[i] = None  # Undo the move

    # Check for a blocking move
    for i in range(9):
        if new_board[i] == None:
            new_board[i] = player_symbol
            if check_winner(convert_to_matrix(new_board)) == player_symbol:
                return i
            new_board[i] = None  # Undo the move

    # Try to take the center
    if new_board[4] == None:
        return 4

    # Try to take a corner
    corners = [0, 2, 6, 8]
    random.shuffle(corners)
    for corner in corners:
        if new_board[corner] == None:
            return corner

    # Take any available side
    sides = [1, 3, 5, 7]
    random.shuffle(sides)
    for side in sides:
        if new_board[side] == None:
            return side

    # If all else fails, return a random available move
    available_moves = [i for i in range(9) if new_board[i] == None]
    if available_moves:
        return random.choice(available_moves)
    else:
        return None

def convert_to_matrix(board):
    return [board[i:i+3] for i in range(0, len(board), 3)]

def check_winner(board):
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != None:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != None:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != None:
        return board[0][2]
    return None