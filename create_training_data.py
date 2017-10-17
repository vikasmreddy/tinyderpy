import copy
import sys
from enum import Enum

class Player(Enum):
    X = 1
    O = 2

def check_won(board_state):

    #check rows
    for column in range(0,3):
        if (board_state[0][column] == board_state[1][column] == board_state[2][column]):
            return board_state[0][column]

    #check columns
    for row in range(0,3):
        if (board_state[row][0] == board_state[row][1] == board_state[row][2]):
            return board_state[row][0]

    #check diagonals
    if board_state[0][0] == board_state[1][1] == board_state[2][2]:
        return board_state[0][0]
    elif board_state[0][2] == board_state[1][1] == board_state[2][0]:
        return board_state[0][2]

    return None

def get_empty_squares(board_state):
    empty_squares = []
    for row in range(0,3):
        for column in range(0,3):
            if board_state[row][column] == None:
                empty_squares.append((row,column))

    return empty_squares

def get_opponent(whose_turn):
    return Player.O if whose_turn == Player.X else Player.X

def score_board(board_state, whose_turn):
    opponent = get_opponent(whose_turn)

    board_win_status = check_won(board_state)

    if board_win_status == whose_turn:
        return 1
    elif board_win_status == opponent:
        return -1

    #if no more empty squares, and neither player won then it's a tie
    if len(get_empty_squares(board_state)) == 0:
        return 0
    else:
        return None

def minimax(board_state, whose_turn, whose_perspective):
    score = score_board(board_state, whose_perspective)

    if score != None:
        return (None, score)

    if whose_turn == whose_perspective:
        best_score = -99
        best_move = None
    else:
        best_score = 99
        best_move = None

    for empty_square in get_empty_squares(board_state):
        modified_board = copy.deepcopy(board_state)
        modified_board[empty_square[0]][empty_square[1]] = whose_turn

        modified_board_minimax_result = minimax(modified_board, get_opponent(whose_turn), whose_perspective)

        if whose_turn == whose_perspective:
            if modified_board_minimax_result[1] > best_score:
                best_move = empty_square
                best_score = modified_board_minimax_result[1]
        else:
            if modified_board_minimax_result[1] < best_score:
                best_move = empty_square
                best_score = modified_board_minimax_result[1]

    return (best_move, best_score)

def output_move_to_regressed_value(move):
    binary_vector_one_hot = [0]*9
    binary_vector_one_hot[move[0]*3+move[1]] = 1
    return binary_vector_one_hot

def binary_vector_one_hot_to_move_to_index(binary_vector_one_hot):
    index = -10000
    for i in range(0,9):
        if binary_vector_one_hot[i] == 1:
            index = i

    if index < 0:
        raise Error("yo")

    return index

def regressed_value_to_move(index):
    return (index/3, index%3)

def board_state_to_vector(board_state, whose_turn):
    binary_vector = [0]*19

    if whose_turn == Player.X:
        binary_vector[18] = 0
    else:
        binary_vector[18] = 1

    for row in range(0,3):
        for col in range(0,3):
            index = (row*3 + col) * 2

            if board_state[row][col] == Player.X:
                binary_vector[index] = 1
            elif board_state[row][col] == Player.O:
                binary_vector[index+1] = 1
            else:
                pass # do nothing here since both spots should be 0

    return binary_vector

def vector_to_board_state_and_whose_turn(binary_vector):
    board_state = [
        [None, None, None],
        [None, None, None],
        [None, None, None]
    ]

    for index in range(0,9):
        board_pos = (index / 3, index % 3)

        filled_square = None

        if binary_vector[index*2] == 1:
            filled_square = Player.X
        elif binary_vector[index*2+1] == 1:
            filled_square = Player.O
        else:
            pass #do nothing here since we initialized to None anyways

        board_state[board_pos[0]][board_pos[1]] = filled_square

    whose_turn = None
    if binary_vector[18] == 1:
        whose_turn = Player.X
    else:
        whose_turn = Player.O

    return board_state, whose_turn


def generate_training_example(initial_board_state, whose_turn, whose_perspective):
    result = minimax(initial_board_state, whose_turn, whose_perspective)

    move = result[0]
    if move == None:
        return None

    binary_input_vector = board_state_to_vector(initial_board_state, whose_turn)
    output_scalar = output_move_to_regressed_value(move)

    return (binary_input_vector, output_scalar, move)

def print_board_state(board_state):
    for row in range(0,3):
        for col in range(0,3):
            if board_state[row][col] == Player.X:
                sys.stdout.write('X')
            elif board_state[row][col] == Player.O:
                sys.stdout.write('O')
            else:
                sys.stdout.write('Z')
        sys.stdout.write('\n')
    sys.stdout.flush()

def create_training_example_generator():
    for row in range(0,3):
        for col in range(0,3):
            #print "********** ONE PATH ***********"
            board_state = [
                [None, None, None],
                [None, None, None],
                [None, None, None]
            ]

            board_state[row][col] = Player.X

            whose_turn = Player.O
            while True:
                #print_board_state(board_state)
                training_example = generate_training_example(board_state, whose_turn, whose_turn)

                if training_example == None:
                    break

                #print training_example

                move = training_example[2]
                board_state[move[0]][move[1]] = whose_turn

                if whose_turn == Player.X:
                    whose_turn = Player.O
                else:
                    whose_turn = Player.X

                yield training_example



# initial_board_state = [
#     [None, Player.O, Player.X],
#     [Player.O, None, Player.X],
#     [Player.O, Player.X, Player.O]
# ]

# initial_board_state = [
#     [Player.X, Player.O, Player.X],
#     [Player.O, Player.X, Player.X],
#     [Player.O, Player.X, Player.O]
# ]

# initial_board_state = [
#     [Player.X, Player.X, Player.O],
#     [None, Player.X, None],
#     [Player.X, Player.O, None]
# ]

#empty_squares = get_empty_squares(initial_board_state)
#print empty_squares

#result = minimax(initial_board_state, Player.X, Player.X)


#training_example = generate_training_example(initial_board_state, Player.X, Player.X)


#training_example_generator = create_training_example_generator()

#for training_example in training_example_generator:
#    print training_example

#print training_example

# 3 * 3 * 2 = 18 (is an X or O in a square) + 1 (whose turn is it) binary input vector
#
# Ax + b =
#
#
# [0,1,2
#  3,4,5
#  6,7,8]
