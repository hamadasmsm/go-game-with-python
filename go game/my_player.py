from minimax import *


def run_human():
    previous_state = np.zeros((9, 9), dtype=np.int)
    current_state = np.zeros((9, 9), dtype=np.int)
    board_AI = Board(1, previous_state, current_state, 0, (10, -2, 6))
    board_me = Board(2, previous_state, board_AI.state, 0, 3)

    max_moves = 0
    is_pass = 0

    while is_pass != 2 and max_moves < 80:
        board_AI.num_moves = max_moves
        if board_AI.num_moves > 9:
            minmax.DEPTH = 4
        v, move = minmax.minimax(board_AI, True, MIN, MAX, 0)

        if move == "PASS":
            print("AI move: ", move)
            print(board_AI.state)
            board_me.previous_board = np.copy(board_AI.state)
            board_me.state = np.copy(board_AI.state)
            board_me.state_copy = np.copy(board_AI.state)
            is_pass += 1
        else:
            previous_state = np.copy(board_AI.state)
            board_AI.my_move(move)

            print("AI move: ", move)
            print(board_AI.state)

            board_me.previous_board = np.copy(previous_state)
            board_me.state = np.copy(board_AI.state)
            board_me.state_copy = np.copy(board_AI.state)
            is_pass = 0

        max_moves += 1

        if max_moves == 80:
            break

        if is_pass == 2:
            break

        human = input("Enter your move: ")

        if human == "PASS":
            print("Human move: ", human)
            print(board_me.state)
            board_AI.previous_board = np.copy(board_me.state)
            board_AI.state = np.copy(board_me.state)
            board_AI.state_copy = np.copy(board_me.state)
            is_pass += 1
        else:
            human = human.split(",")
            x = int(human[0])
            y = int(human[1])
            previous_state = np.copy(board_me.state)
            board_me.my_move(tuple([x, y]))

            print("Human move: ", human)
            print(board_me.state)

            board_AI.previous_board = np.copy(previous_state)
            board_AI.state = np.copy(board_me.state)
            board_AI.state_copy = np.copy(board_me.state)
            is_pass = 0
        max_moves += 1

    print("AI: ", board_AI.count_stones())
    print("Me: ", board_me.count_stones())


if __name__ == "__main__":
    minmax = MinMax()
    # if human is white
    run_human()
