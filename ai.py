from board import Board
from copy import deepcopy
from random import choice
import time

class AI:
    def __init__(self, color):
        # 'color' is the color this AI will play with (B or W)
        self.color = color
        self.transposition_table = {}  # Cache for board positions
        self.MAX_DEPTH = 6  # Increased from 2 to 6 for better lookahead
        
    def minimax_alpha_beta(self, current_board, is_maximizing, depth, turn, alpha, beta):
        """
        Enhanced minimax with alpha-beta pruning for better performance
        """
        # Check if we've seen this position before (transposition table)
        board_hash = self._get_board_hash(current_board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash]['depth'] >= depth:
            return self.transposition_table[board_hash]['value']
        
        # Terminal node evaluation
        if depth == 0 or current_board.get_winner() is not None:
            value = self.get_advanced_value(current_board, depth)
            self.transposition_table[board_hash] = {'value': value, 'depth': depth}
            return value
        
        next_turn = 'B' if turn == 'W' else 'W'
        board_color_up = current_board.get_color_up()
        current_pieces = current_board.get_pieces()
        
        # Get all possible moves
        all_moves = []
        for index, piece in enumerate(current_pieces):
            if piece.get_color() == turn:
                moves = piece.get_moves(current_board)
                for move in moves:
                    all_moves.append((index, move))
        
        # Order moves for better pruning (captures first)
        all_moves.sort(key=lambda x: x[1]["eats_piece"], reverse=True)
        
        if is_maximizing:
            max_eval = -999999
            for index, move in all_moves:
                aux_board = Board(deepcopy(current_pieces), board_color_up)
                aux_board.move_piece(index, int(move["position"]))
                
                # Check for multiple jumps
                if move["eats_piece"] and self._can_continue_jumping(aux_board, index):
                    eval = self.minimax_alpha_beta(aux_board, True, depth - 1, turn, alpha, beta)
                else:
                    eval = self.minimax_alpha_beta(aux_board, False, depth - 1, next_turn, alpha, beta)
                
                max_eval = max(eval, max_eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            
            self.transposition_table[board_hash] = {'value': max_eval, 'depth': depth}
            return max_eval
        else:
            min_eval = 999999
            for index, move in all_moves:
                aux_board = Board(deepcopy(current_pieces), board_color_up)
                aux_board.move_piece(index, int(move["position"]))
                
                # Check for multiple jumps
                if move["eats_piece"] and self._can_continue_jumping(aux_board, index):
                    eval = self.minimax_alpha_beta(aux_board, False, depth - 1, turn, alpha, beta)
                else:
                    eval = self.minimax_alpha_beta(aux_board, True, depth - 1, next_turn, alpha, beta)
                
                min_eval = min(eval, min_eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            self.transposition_table[board_hash] = {'value': min_eval, 'depth': depth}
            return min_eval
    
    def _can_continue_jumping(self, board, piece_index):
        """Check if a piece can make another jump after eating"""
        piece = board.get_pieces()[piece_index]
        moves = piece.get_moves(board)
        return any(move["eats_piece"] for move in moves)
    
    def _get_board_hash(self, board):
        """Generate a hash for the board position for transposition table"""
        pieces_str = ""
        for piece in board.get_pieces():
            pieces_str += piece.get_name()
        return hash(pieces_str)
    
    def get_advanced_value(self, board, depth):
        """
        Advanced evaluation function considering multiple strategic factors
        """
        board_pieces = board.get_pieces()
        winner = board.get_winner()
        
        # Terminal states with depth bonus (prefer quicker wins)
        if winner is not None:
            if winner == self.color:
                return 10000 + depth * 10  # Win faster is better
            else:
                return -10000 - depth * 10  # Lose slower is better
        
        # Count pieces and their values
        player_regular = 0
        player_kings = 0
        opponent_regular = 0
        opponent_kings = 0
        
        # Positional values
        player_position_value = 0
        opponent_position_value = 0
        
        # Advanced pieces bonus
        player_advanced_pieces = 0
        opponent_advanced_pieces = 0
        
        # Protected pieces (pieces that can't be captured immediately)
        player_protected = 0
        opponent_protected = 0
        
        for piece in board_pieces:
            position = int(piece.get_position())
            row = board.get_row_number(position)
            col = board.get_col_number(position)
            
            if piece.get_color() == self.color:
                if piece.is_king():
                    player_kings += 1
                else:
                    player_regular += 1
                
                # Positional evaluation
                player_position_value += self._evaluate_position(piece, row, col, board)
                
                # Advanced pieces (non-kings that are close to becoming kings)
                if not piece.is_king():
                    if (board.get_color_up() == self.color and row <= 2) or \
                       (board.get_color_up() != self.color and row >= 5):
                        player_advanced_pieces += 1
                
                # Check if piece is protected
                if self._is_protected(piece, board):
                    player_protected += 1
            else:
                if piece.is_king():
                    opponent_kings += 1
                else:
                    opponent_regular += 1
                
                # Positional evaluation
                opponent_position_value += self._evaluate_position(piece, row, col, board)
                
                # Advanced pieces
                if not piece.is_king():
                    if (board.get_color_up() != self.color and row <= 2) or \
                       (board.get_color_up() == self.color and row >= 5):
                        opponent_advanced_pieces += 1
                
                # Check if piece is protected
                if self._is_protected(piece, board):
                    opponent_protected += 1
        
        # Calculate final score with weighted factors
        score = 0
        
        # Material value (kings worth more)
        score += (player_regular * 100 + player_kings * 150)
        score -= (opponent_regular * 100 + opponent_kings * 150)
        
        # Position value
        score += player_position_value * 2
        score -= opponent_position_value * 2
        
        # Advanced pieces bonus
        score += player_advanced_pieces * 20
        score -= opponent_advanced_pieces * 20
        
        # Protection bonus
        score += player_protected * 10
        score -= opponent_protected * 10
        
        # Mobility (number of possible moves)
        player_mobility = self._count_mobility(board, self.color)
        opponent_color = 'W' if self.color == 'B' else 'B'
        opponent_mobility = self._count_mobility(board, opponent_color)
        score += player_mobility * 5
        score -= opponent_mobility * 5
        
        # Back row protection bonus (keeping pieces on back row early game)
        if len(board_pieces) > 16:  # Early/mid game
            score += self._back_row_bonus(board) * 15
        
        return score
    
    def _evaluate_position(self, piece, row, col, board):
        """Evaluate positional strength of a piece"""
        value = 0
        
        # Center control is valuable
        if 2 <= col <= 5:
            value += 3
        if 2 <= row <= 5:
            value += 3
        
        # Kings are more valuable in the center
        if piece.is_king():
            if 2 <= row <= 5 and 2 <= col <= 5:
                value += 5
        
        # Edge pieces are slightly less valuable (except back row)
        if col == 0 or col == 7:
            value -= 2
        
        return value
    
    def _is_protected(self, piece, board):
        """Check if a piece is protected (can't be captured immediately)"""
        position = int(piece.get_position())
        row = board.get_row_number(position)
        col = board.get_col_number(position)
        
        # Back row pieces are always protected
        if (piece.get_color() == board.get_color_up() and row == 7) or \
           (piece.get_color() != board.get_color_up() and row == 0):
            return True
        
        # Edge pieces have some protection
        if col == 0 or col == 7:
            return True
        
        return False
    
    def _count_mobility(self, board, color):
        """Count total number of moves available for a color"""
        total_moves = 0
        for piece in board.get_pieces():
            if piece.get_color() == color:
                total_moves += len(piece.get_moves(board))
        return total_moves
    
    def _back_row_bonus(self, board):
        """Bonus for keeping pieces on the back row in early game"""
        bonus = 0
        for piece in board.get_pieces():
            if piece.get_color() == self.color and not piece.is_king():
                position = int(piece.get_position())
                row = board.get_row_number(position)
                if (board.get_color_up() == self.color and row == 7) or \
                   (board.get_color_up() != self.color and row == 0):
                    bonus += 1
        return bonus
    
    def minimax(self, current_board, is_maximizing, depth, turn):
        """Backward compatibility wrapper - uses alpha-beta internally"""
        return self.minimax_alpha_beta(current_board, is_maximizing, depth, turn, -999999, 999999)
    
    def get_move(self, current_board):
        """Get the best move using iterative deepening and time management"""
        board_color_up = current_board.get_color_up()
        current_pieces = current_board.get_pieces()
        next_turn = "W" if self.color == "B" else "B"
        player_pieces = list(map(lambda piece: piece if piece.get_color() == self.color else False, current_pieces))
        possible_moves = []
        
        # Clear transposition table if it gets too large
        if len(self.transposition_table) > 10000:
            self.transposition_table.clear()
        
        for index, piece in enumerate(player_pieces):
            if piece == False:
                continue
            
            for move in piece.get_moves(current_board):
                possible_moves.append({"piece": index, "move": move})
        
        # If any jump move is available, only jump moves can be made
        jump_moves = list(filter(lambda move: move["move"]["eats_piece"] == True, possible_moves))
        if len(jump_moves) != 0:
            possible_moves = jump_moves
        
        # If only one move possible, return it immediately
        if len(possible_moves) == 1:
            move_chosen = possible_moves[0]
            return {"position_to": move_chosen["move"]["position"], 
                   "position_from": player_pieces[move_chosen["piece"]].get_position()}
        
        # Iterative deepening with time limit
        best_move = None
        start_time = time.time()
        time_limit = 2.0  # 2 seconds max thinking time
        
        for depth in range(2, self.MAX_DEPTH + 1):
            if time.time() - start_time > time_limit:
                break
            
            move_scores = []
            for move in possible_moves:
                aux_board = Board(deepcopy(current_pieces), board_color_up)
                aux_board.move_piece(move["piece"], int(move["move"]["position"]))
                
                # Check for multiple jumps
                if move["move"]["eats_piece"] and self._can_continue_jumping(aux_board, move["move"]["position"]):
                    score = self.minimax_alpha_beta(aux_board, True, depth, self.color, -999999, 999999)
                else:
                    score = self.minimax_alpha_beta(aux_board, False, depth, next_turn, -999999, 999999)
                
                move_scores.append(score)
            
            # Find best moves at this depth
            best_score = max(move_scores)
            best_moves_at_depth = []
            for index, move in enumerate(possible_moves):
                if move_scores[index] == best_score:
                    best_moves_at_depth.append(move)
            
            # Update best move
            best_move = choice(best_moves_at_depth)
        
        return {"position_to": best_move["move"]["position"], 
               "position_from": player_pieces[best_move["piece"]].get_position()}
    
    def get_value(self, board):
        """Backward compatibility - redirects to advanced evaluation"""
        return self.get_advanced_value(board, 0)