# from read import readInput
# from write import writeOutput
# from host import GO

# class MyPlayer():
#     def __init__(self):
#         self.type = 'my_player'

#     def get_input(self, go, piece_type):
#         '''
#         Get one input.

#         :param go: Go instance.
#         :param piece_type: 1('X') or 2('O').
#         :return: (row, column) coordinate of input.
#         '''
#         # TODO: Implement your strategy here
#         # For now, we'll use a simple random strategy
#         possible_placements = []
#         for i in range(go.size):
#             for j in range(go.size):
#                 if go.valid_place_check(i, j, piece_type, test_check=True):
#                     possible_placements.append((i,j))

#         if not possible_placements:
#             return "PASS"
#         else:
#             return random.choice(possible_placements)

# if __name__ == "__main__":
#     N = 5
#     piece_type, previous_board, board = readInput(N)
#     go = GO(N)
#     go.set_board(piece_type, previous_board, board)
#     player = MyPlayer()
#     action = player.get_input(go, piece_type)
#     writeOutput(action)

from read import readInput
from write import writeOutput
from host import GO
import random

class MyPlayer():
    def __init__(self):
        self.type = 'my_player'
        self.max_depth = 4  # Adjust this value based on performance

    def get_input(self, go, piece_type):
        '''
        Get one input.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: (row, column) coordinate of input or "PASS".
        '''
        possible_placements = self.get_possible_placements(go, piece_type)
        
        if not possible_placements:
            return "PASS"
        
        best_move = self.minimax(go, piece_type, self.max_depth, float('-inf'), float('inf'), True)[1]
        return best_move

    def get_possible_placements(self, go, piece_type):
        '''
        Get all possible placements for the current board state.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: List of possible placements.
        '''
        possible_placements = []
        for i in range(go.size):
            for j in range(go.size):
                if go.valid_place_check(i, j, piece_type, test_check=True):
                    possible_placements.append((i,j))
        return possible_placements

    def minimax(self, go, piece_type, depth, alpha, beta, maximizing_player):
        '''
        Minimax algorithm with alpha-beta pruning.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :param depth: Current depth in the game tree.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :param maximizing_player: Boolean indicating if current player is maximizing.
        :return: (score, best_move)
        '''
        if depth == 0 or go.game_end(piece_type):
            return self.evaluate(go, piece_type), None

        possible_placements = self.get_possible_placements(go, piece_type)
        
        if not possible_placements:
            return self.evaluate(go, piece_type), "PASS"

        best_move = random.choice(possible_placements)  # Default to random move

        if maximizing_player:
            max_eval = float('-inf')
            for move in possible_placements:
                new_go = go.copy_board()
                new_go.place_chess(move[0], move[1], piece_type)
                new_go.remove_died_pieces(3 - piece_type)
                eval = self.minimax(new_go, 3 - piece_type, depth - 1, alpha, beta, False)[0]
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in possible_placements:
                new_go = go.copy_board()
                new_go.place_chess(move[0], move[1], piece_type)
                new_go.remove_died_pieces(3 - piece_type)
                eval = self.minimax(new_go, 3 - piece_type, depth - 1, alpha, beta, True)[0]
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def evaluate(self, go, piece_type):
        '''
        Evaluate the current board state.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: Score for the current board state.
        '''
        my_score = go.score(piece_type)
        opponent_score = go.score(3 - piece_type)
        
        # Consider liberty
        my_liberty = self.count_liberty(go, piece_type)
        opponent_liberty = self.count_liberty(go, 3 - piece_type)
        
        # Consider board control (empty adjacent points)
        my_control = self.count_control(go, piece_type)
        opponent_control = self.count_control(go, 3 - piece_type)
        
        return (my_score - opponent_score) + 0.5 * (my_liberty - opponent_liberty) + 0.3 * (my_control - opponent_control)

    def count_liberty(self, go, piece_type):
        '''
        Count the total liberty for a player.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: Total liberty count.
        '''
        liberty_count = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    liberty_count += len(go.detect_neighbor_ally(i, j))
        return liberty_count

    def count_control(self, go, piece_type):
        '''
        Count the number of empty adjacent points for a player.

        :param go: Go instance.
        :param piece_type: 1('X') or 2('O').
        :return: Control count.
        '''
        control_count = 0
        for i in range(go.size):
            for j in range(go.size):
                if go.board[i][j] == piece_type:
                    neighbors = go.detect_neighbor(i, j)
                    control_count += sum(1 for x, y in neighbors if go.board[x][y] == 0)
        return control_count

if __name__ == "__main__":
    N = 5
    piece_type, previous_board, board = readInput(N)
    go = GO(N)
    go.set_board(piece_type, previous_board, board)
    player = MyPlayer()
    action = player.get_input(go, piece_type)
    writeOutput(action)