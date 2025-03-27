import numpy as np
from copy import deepcopy

class MyPlayer:
    def __init__(self):
        self.board_size = 5
        self.piece_type = None
        self.previous_board = None
        self.board = None
        self.max_depth = 3  # Adjust based on time constraints
        self.valid_moves_cache = {}
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self):
        patterns = {
            'corner': [(0,0), (0,4), (4,0), (4,4)],  # Corners are valuable
            'edge': [(0,1), (0,2), (0,3), (1,0), (1,4), (2,0), (2,4), (3,0), (3,4), (4,1), (4,2), (4,3)],
            'center': [(2,2)],  # Center control
            'near_center': [(1,1), (1,2), (1,3), (2,1), (2,3), (3,1), (3,2), (3,3)]
        }
        return patterns

    def get_input(self, piece_type, previous_board, board):
        """Get input and store game state"""
        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
        
    def detect_neighbor_ally(self, i, j, board, piece_type):
        """Detect ally neighbors of a position"""
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                if board[ni][nj] == piece_type:
                    neighbors.append((ni, nj))
        return neighbors

    def detect_neighbor_empty(self, i, j, board):
        """Detect empty neighbors of a position"""
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                if board[ni][nj] == 0:
                    neighbors.append((ni, nj))
        return neighbors

    def find_liberty(self, i, j, board, piece_type, visited=None):
        """Find liberties of a group"""
        if visited is None:
            visited = set()
        
        queue = [(i,j)]
        visited.add((i,j))
        liberty = set()
        while queue:
            curr_i, curr_j = queue.pop(0)
            empty_neighbors = self.detect_neighbor_empty(curr_i, curr_j, board)
            liberty.update(empty_neighbors)
            
            ally_neighbors = self.detect_neighbor_ally(curr_i, curr_j, board, piece_type)
            for ni, nj in ally_neighbors:
                if (ni, nj) not in visited:
                    queue.append((ni, nj))
                    visited.add((ni, nj))
        
        return liberty

    def find_all_dead_pieces(self, piece_type, board):
        """Find all dead pieces on board"""
        dead_pieces = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == piece_type:
                    if not self.find_liberty(i, j, board, piece_type):
                        dead_pieces.append((i,j))
        return dead_pieces

    def remove_dead_pieces(self, piece_type, board):
        """Remove dead pieces from board"""
        dead_pieces = self.find_all_dead_pieces(piece_type, board)
        for i, j in dead_pieces:
            board[i][j] = 0
        return board, len(dead_pieces)

    def valid_moves(self, piece_type, board):
        """Get all valid moves"""
        moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == 0:
                    if self.valid_place_check(i, j, piece_type, board):
                        moves.append((i,j))
        return moves

    def valid_place_check(self, i, j, piece_type, board):
        """Check if a move is valid"""
        # Check if position is empty
        if board[i][j] != 0:
            return False
            
        test_board = deepcopy(board)
        test_board[i][j] = piece_type
        
        # Check for ko rule
        if test_board == self.previous_board:
            return False
            
        # Check for liberty rule
        if not self.find_liberty(i, j, test_board, piece_type):
            # Check if move captures enemy pieces
            enemy_type = 3 - piece_type
            dead_pieces = self.find_all_dead_pieces(enemy_type, test_board)
            if not dead_pieces:
                return False
                
        return True

    def evaluate_board(self, board, piece_type):
        """Evaluate board state"""
        score = 0
        enemy_type = 3 - piece_type
        
        # Count stones and territories
        my_stones = 0
        enemy_stones = 0
        my_liberties = 0
        enemy_liberties = 0
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == piece_type:
                    my_stones += 1
                    my_liberties += len(self.find_liberty(i, j, board, piece_type))
                    # Bonus for pattern positions
                    if (i,j) in self.patterns['corner']:
                        score += 3
                    elif (i,j) in self.patterns['edge']:
                        score += 2
                    elif (i,j) in self.patterns['center']:
                        score += 4
                    elif (i,j) in self.patterns['near_center']:
                        score += 1
                elif board[i][j] == enemy_type:
                    enemy_stones += 1
                    enemy_liberties += len(self.find_liberty(i, j, board, enemy_type))
        
        # Weight different factors
        score += (my_stones - enemy_stones) * 10
        score += (my_liberties - enemy_liberties) * 0.5
        
        return score

    def minimax(self, board, depth, alpha, beta, maximizing_player, piece_type):
        """Minimax algorithm with alpha-beta pruning"""
        if depth == 0:
            return self.evaluate_board(board, self.piece_type), None
            
        moves = self.valid_moves(piece_type, board)
        if not moves:
            return self.evaluate_board(board, self.piece_type), None
            
        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            for i, j in moves:
                new_board = deepcopy(board)
                new_board[i][j] = piece_type
                new_board, _ = self.remove_dead_pieces(3 - piece_type, new_board)
                
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, False, 3 - piece_type)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = (i,j)
                    
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
                    
            return max_eval, best_move
            
        else:
            min_eval = float('inf')
            best_move = None
            for i, j in moves:
                new_board = deepcopy(board)
                new_board[i][j] = piece_type
                new_board, _ = self.remove_dead_pieces(3 - piece_type, new_board)
                
                eval, _ = self.minimax(new_board, depth - 1, alpha, beta, True, 3 - piece_type)
                
                if eval < min_eval:
                    min_eval = eval
                    best_move = (i,j)
                    
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                    
            return min_eval, best_move

    def get_move(self, piece_type, previous_board, board):
        """Get next move"""
        self.get_input(piece_type, previous_board, board)
        
        # If this is the first move, play near center
        total_pieces = np.sum(board != 0)
        if total_pieces <= 1:
            moves = [(2,2), (1,1), (1,3), (3,1), (3,3)]
            for i, j in moves:
                if self.valid_place_check(i, j, piece_type, board):
                    return i, j
        
        # Check if we should pass
        valid_moves = self.valid_moves(piece_type, board)
        if not valid_moves:
            return "PASS"
        
        # Use minimax to find best move
        _, best_move = self.minimax(board, self.max_depth, float('-inf'), float('inf'), True, piece_type)
        
        if best_move:
            return best_move
        else:
            # Fallback to first valid move
            return valid_moves[0]

    def write_output(self, result):
        """Write output to file"""
        if result == "PASS":
            with open("output.txt", 'w') as f:
                f.write("PASS")
        else:
            with open("output.txt", 'w') as f:
                f.write(f"{result[0]},{result[1]}")

def readInput(n, path="input.txt"):
    """Read input from file"""
    with open(path, 'r') as f:
        lines = f.readlines()
        
    piece_type = int(lines[0])
    previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
    board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1:2*n+1]]
    
    return piece_type, previous_board, board

def main():
    """Main function"""
    N = 5
    piece_type, previous_board, board = readInput(N)
    player = MyPlayer()
    result = player.get_move(piece_type, previous_board, board)
    player.write_output(result)

if __name__ == '__main__':
    main()