import tkinter as tk
from tkinter import messagebox
import copy
import math
import random
import time
from abc import ABC, abstractmethod
from enum import Enum

class PlayerType(Enum):
    GREEDY = "Greedy Bot"
    MINIMAX_SIMPLE = "Simple Minimax"
    MINIMAX_SMART = "Smart Minimax"
    RANDOM = "Random Bot"

# --- Konfiguration für Remis ---
THREEFOLD_REPETITION = 3
HALFMOVE_DRAW_LIMIT = 200  # vorher 100
DRAW_BIAS = 10.0            # Draw ist so „schlecht“ für die Seite mit Vorteil

class GameStats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.moves_count = 0
        self.game_start_time = time.time()
        self.positions_evaluated = 0
        self.pruning_cuts = 0
        self.move_times = []
        self.captured_pieces = {1: 0, -1: 0}
        self.first_king_time = {1: None, -1: None}

class BaseAI(ABC):
    def __init__(self, player, name):
        self.player = player
        self.name = name
        self.stats = GameStats()
    @abstractmethod
    def get_move(self, game):
        pass
    def reset_stats(self):
        self.stats.reset()

class RandomAI(BaseAI):
    def __init__(self, player):
        super().__init__(player, "Random Bot")
    def get_move(self, game):
        start_time = time.time()
        moves = game.get_all_moves(self.player)
        if moves:
            move = random.choice(moves)
            self.stats.move_times.append(time.time() - start_time)
            return move
        return None

class GreedyAI(BaseAI):
    def __init__(self, player):
        super().__init__(player, "Greedy Bot (Depth 1)")
    def get_move(self, game):
        start_time = time.time()
        moves = game.get_all_moves(self.player)
        if not moves:
            return None
        best_score = -math.inf
        best_move = moves[0]
        for move in moves:
            new_board = game.simulate_complete_move(game.board, move)
            score = self.evaluate_board(new_board, game)
            self.stats.positions_evaluated += 1
            if move['captured']:
                score += 15
            if score > best_score:
                best_score = score
                best_move = move
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    def evaluate_board(self, board, game):
        score = 0
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0:
                    piece_value = 10 if abs(piece) == 1 else 50
                    score += piece_value if piece * self.player > 0 else -piece_value
        return score

class SimpleMinimaxAI(BaseAI):
    def __init__(self, player, depth=2):
        super().__init__(player, f"Simple Minimax (Depth {depth}, No Pruning)")
        self.depth = depth
    def get_move(self, game):
        start_time = time.time()
        _, best_move = self.minimax(game, game.board, self.depth, self.player == -1, game.halfmove_clock)
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    def minimax(self, game, board, depth, maximizing_player, halfmove_clock):
        # 50-Züge-Regel (suche-aware)
        if halfmove_clock >= HALFMOVE_DRAW_LIMIT:
            self.stats.positions_evaluated += 1
            # Draw „bestrafen“ für die Seite mit Materialvorteil
            bias = DRAW_BIAS * (0.5 + halfmove_clock / HALFMOVE_DRAW_LIMIT)  # steigt, je näher am 50-Züge-Draw
            draw_score = -bias * game.material_balance(board)
            return draw_score, None
        if depth == 0:
            score = game.evaluate_board_static(board)
            self.stats.positions_evaluated += 1
            return score, None
        player = -1 if maximizing_player else 1
        possible_moves = game.get_all_moves_for_board(board, player)
        if not possible_moves:
            self.stats.positions_evaluated += 1
            return (-1000 if maximizing_player else 1000), None
        best_move = None
        if maximizing_player:
            max_eval = -math.inf
            for move in possible_moves:
                new_board, progress = game.simulate_complete_move_with_progress(board, move)
                new_clock = 0 if progress else halfmove_clock + 1
                if new_clock >= HALFMOVE_DRAW_LIMIT:
                    eval_score = 0
                else:
                    eval_score, _ = self.minimax(game, new_board, depth - 1, False, new_clock)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in possible_moves:
                new_board, progress = game.simulate_complete_move_with_progress(board, move)
                new_clock = 0 if progress else halfmove_clock + 1
                if new_clock >= HALFMOVE_DRAW_LIMIT:
                    eval_score = 0
                else:
                    eval_score, _ = self.minimax(game, new_board, depth - 1, True, new_clock)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            return min_eval, best_move


class SmartMinimaxAI(BaseAI):
    def __init__(self, player, depth=5):
        super().__init__(player, f"Smart Minimax (Depth {depth}, Alpha-Beta)")
        self.depth = depth
    def get_move(self, game):
        start_time = time.time()
        _, best_move = self.minimax(game, game.board, self.depth, -math.inf, math.inf, self.player == -1, game.halfmove_clock)
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    def minimax(self, game, board, depth, alpha, beta, maximizing_player, halfmove_clock):
        if halfmove_clock >= HALFMOVE_DRAW_LIMIT:
            self.stats.positions_evaluated += 1
            bias = DRAW_BIAS * (0.5 + halfmove_clock / HALFMOVE_DRAW_LIMIT)
            draw_score = -bias * game.material_balance(board)
            return draw_score, None
        if depth == 0:
            score = game.evaluate_board_static(board)
            self.stats.positions_evaluated += 1
            return score, None
        player = -1 if maximizing_player else 1
        possible_moves = game.get_all_moves_for_board(board, player)
        if not possible_moves:
            self.stats.positions_evaluated += 1
            return (-1000 if maximizing_player else 1000), None
        best_move = None
        if maximizing_player:
            max_eval = -math.inf
            for move in possible_moves:
                new_board, progress = game.simulate_complete_move_with_progress(board, move)
                new_clock = 0 if progress else halfmove_clock + 1
                eval_score = 0 if new_clock >= HALFMOVE_DRAW_LIMIT else \
                    self.minimax(game, new_board, depth - 1, alpha, beta, False, new_clock)[0]
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.stats.pruning_cuts += 1
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in possible_moves:
                new_board, progress = game.simulate_complete_move_with_progress(board, move)
                new_clock = 0 if progress else halfmove_clock + 1
                eval_score = 0 if new_clock >= HALFMOVE_DRAW_LIMIT else \
                    self.minimax(game, new_board, depth - 1, alpha, beta, True, new_clock)[0]
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.stats.pruning_cuts += 1
                    break
            return min_eval, best_move

class CheckersGame:
    def __init__(self, mode="AIvAI", red_ai_type=None, black_ai_type=None, gui=True):
        self.board = self.create_initial_board()
        self.current_player = 1
        self.selected_piece = None
        self.selected_pos = None
        self.must_jump = False
        self.jump_piece = None
        self.mode = mode
        self.gui = gui
        self.game_over = False

        # Remis-Tracking
        self.halfmove_clock = 0
        self.position_history = {}

        # Statistiken
        self.game_stats = GameStats()

        # KI-Spieler erstellen
        self.red_ai = self.create_ai(1, red_ai_type) if red_ai_type else None
        self.black_ai = self.create_ai(-1, black_ai_type) if black_ai_type else None

        # Startstellung in History eintragen
        self.record_position()

        if gui:
            self.setup_gui()



    # --- Remis-Helfer ---
    def position_key(self, board=None, player_to_move=None):
        b = board if board is not None else self.board
        p = self.current_player if player_to_move is None else player_to_move
        return (p, tuple(tuple(row) for row in b))

    def record_position(self):
        key = self.position_key()
        self.position_history[key] = self.position_history.get(key, 0) + 1

    def is_draw_by_repetition(self):
        key = self.position_key()
        return self.position_history.get(key, 0) >= THREEFOLD_REPETITION

    def is_draw_by_50_move_rule(self):
        return self.halfmove_clock >= HALFMOVE_DRAW_LIMIT

    def create_ai(self, player, ai_type):
        if ai_type == PlayerType.RANDOM:
            return RandomAI(player)
        elif ai_type == PlayerType.GREEDY:
            return GreedyAI(player)
        elif ai_type == PlayerType.MINIMAX_SIMPLE:
            return SimpleMinimaxAI(player, depth=2)
        elif ai_type == PlayerType.MINIMAX_SMART:
            return SmartMinimaxAI(player, depth=6)
        return None

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Checkers AI")
        self.root.resizable(False, False)

        main_frame = tk.Frame(self.root)
        main_frame.pack()

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.canvas = tk.Canvas(left_frame, width=640, height=640, bg='white')
        self.canvas.pack()

        self.info_label = tk.Label(left_frame, text="", font=('Arial', 14))
        self.info_label.pack(pady=5)

        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)

        player_frame = tk.LabelFrame(right_frame, text="Current Game", font=('Arial', 12, 'bold'))
        player_frame.pack(fill=tk.X, pady=5)
        self.red_player_label = tk.Label(player_frame, text="", fg="red")
        self.red_player_label.pack()
        self.black_player_label = tk.Label(player_frame, text="", fg="black")
        self.black_player_label.pack()

        stats_frame = tk.LabelFrame(right_frame, text="Live Statistics", font=('Arial', 12, 'bold'))
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_text = tk.Text(stats_frame, width=40, height=16, font=('Courier', 9))
        self.stats_text.pack()

        control_frame = tk.Frame(right_frame)
        control_frame.pack(pady=10)
        self.speed_var = tk.IntVar(value=500)
        tk.Label(control_frame, text="AI Speed (ms):").pack()
        tk.Scale(control_frame, from_=50, to=2000, orient=tk.HORIZONTAL, 
                 variable=self.speed_var, length=200).pack()

        self.pause_var = tk.BooleanVar(value=False)
        tk.Checkbutton(control_frame, text="Pause", variable=self.pause_var).pack()

        self.draw_board()
        self.draw_pieces()
        self.update_display()

    def create_initial_board(self):
        board = [[0 for _ in range(8)] for _ in range(8)]
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = -1
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 1
        return board

    def draw_board(self):
        if not self.gui:
            return
        self.canvas.delete("board")
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 80, row * 80
                x2, y2 = x1 + 80, y1 + 80
                color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="board")

    def draw_pieces(self):
        if not self.gui:
            return
        self.canvas.delete("piece")
        self.canvas.delete("highlight")
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece != 0:
                    x, y = col * 80 + 40, row * 80 + 40
                    if self.selected_pos == (row, col):
                        self.canvas.create_oval(x-35, y-35, x+35, y+35, 
                                                fill='yellow', outline='gold', width=3, tags="highlight")
                    if abs(piece) == 1:
                        color = 'red' if piece == 1 else 'black'
                        outline_color = 'darkred' if piece == 1 else 'gray'
                    else:
                        color = 'red' if piece == 2 else 'black'
                        outline_color = 'gold'
                    self.canvas.create_oval(x-30, y-30, x+30, y+30, 
                                            fill=color, outline=outline_color, width=3, tags="piece")
                    if abs(piece) == 2:
                        self.canvas.create_text(x, y, text="♔", fill='gold', 
                                                font=('Arial', 20, 'bold'), tags="piece")

    def update_display(self):
        if not self.gui:
            return
        if self.current_player == 1:
            player_text = "Red's Turn"
            current_ai = self.red_ai
        else:
            player_text = "Black's Turn"
            current_ai = self.black_ai
        if current_ai:
            player_text += f" ({current_ai.name})"
        self.info_label.config(text=player_text)

        red_text = "Red: " + (self.red_ai.name if self.red_ai else "Human")
        black_text = "Black: " + (self.black_ai.name if self.black_ai else "Human")
        self.red_player_label.config(text=red_text)
        self.black_player_label.config(text=black_text)

        self.update_stats_display()

    def update_stats_display(self):
        if not self.gui:
            return
        stats_text = []
        stats_text.append(f"Move: {self.game_stats.moves_count}")
        red_pieces = sum(1 for row in self.board for piece in row if piece > 0)
        black_pieces = sum(1 for row in self.board for piece in row if piece < 0)
        red_kings = sum(1 for row in self.board for piece in row if piece == 2)
        black_kings = sum(1 for row in self.board for piece in row if piece == -2)
        stats_text.append(f"\nPieces:")
        stats_text.append(f"Red: {red_pieces} ({red_kings} kings)")
        stats_text.append(f"Black: {black_pieces} ({black_kings} kings)")
        stats_text.append(f"\nHalfmove clock: {self.halfmove_clock}/{HALFMOVE_DRAW_LIMIT}")
        if self.red_ai and self.red_ai.stats.positions_evaluated > 0:
            stats_text.append(f"\nRed AI Stats:")
            stats_text.append(f"Positions: {self.red_ai.stats.positions_evaluated}")
            if getattr(self.red_ai, "stats", None) and self.red_ai.stats.pruning_cuts > 0:
                stats_text.append(f"Pruning Cuts: {self.red_ai.stats.pruning_cuts}")
            if self.red_ai.stats.move_times:
                avg_time = sum(self.red_ai.stats.move_times) / len(self.red_ai.stats.move_times)
                stats_text.append(f"Avg Move Time: {avg_time:.3f}s")
        if self.black_ai and self.black_ai.stats.positions_evaluated > 0:
            stats_text.append(f"\nBlack AI Stats:")
            stats_text.append(f"Positions: {self.black_ai.stats.positions_evaluated}")
            if getattr(self.black_ai, "stats", None) and self.black_ai.stats.pruning_cuts > 0:
                stats_text.append(f"Pruning Cuts: {self.black_ai.stats.pruning_cuts}")
            if self.black_ai.stats.move_times:
                avg_time = sum(self.black_ai.stats.move_times) / len(self.black_ai.stats.move_times)
                stats_text.append(f"Avg Move Time: {avg_time:.3f}s")
        if (self.red_ai and self.black_ai and 
            self.red_ai.stats.positions_evaluated > 0 and 
            self.black_ai.stats.positions_evaluated > 0):
            ratio = self.black_ai.stats.positions_evaluated / self.red_ai.stats.positions_evaluated
            stats_text.append(f"\nEfficiency Ratio: {ratio:.2f}")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, '\n'.join(stats_text))

    # --- Such-Helfer für Minimax ---
    def simulate_complete_move_with_progress(self, board, move):
        """Wie simulate_complete_move, gibt zusätzlich zurück,
           ob ein Schlag oder eine Beförderung passierte (progress=True)."""
        new_board = copy.deepcopy(board)
        progress = False
        while True:
            start_row, start_col = move['start']
            end_row, end_col = move['end']
            captured = move['captured']
            piece = new_board[start_row][start_col]
            new_board[end_row][end_col] = piece
            new_board[start_row][start_col] = 0
            if captured:
                cap_row, cap_col = captured
                new_board[cap_row][cap_col] = 0
                progress = True
            # Beförderung?
            if piece == 1 and end_row == 0:
                new_board[end_row][end_col] = 2
                progress = True
            elif piece == -1 and end_row == 7:
                new_board[end_row][end_col] = -2
                progress = True
            # Mehrfachsprung automatisch fortsetzen (gleiche Heuristik wie im Spiel)
            if captured:
                further_jumps = self.get_possible_moves_for_board(new_board, end_row, end_col)
                jump_moves = [m for m in further_jumps if m['captured']]
                if jump_moves:
                    move = jump_moves[0]
                    continue
            break
        return new_board, progress

    def simulate_complete_move(self, board, move):
        # Für Greedy/Evaluationszwecke unverändert
        return self.simulate_complete_move_with_progress(board, move)[0]

    def get_possible_moves(self, row, col):
        piece = self.board[row][col]
        moves = []
        if piece == 0:
            return moves
        is_king = abs(piece) == 2
        player = 1 if piece > 0 else -1
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else (
            [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        )
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == 0:
                moves.append({'start': (row, col), 'end': (new_row, new_col), 'captured': None})
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if (0 <= jump_row < 8 and 0 <= jump_col < 8 and 
                self.board[new_row][new_col] * player < 0 and
                self.board[jump_row][jump_col] == 0):
                moves.append({'start': (row, col), 'end': (jump_row, jump_col), 'captured': (new_row, new_col)})
        return moves

    def has_jumps_available(self, player):
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves(row, col)
                    if any(m['captured'] for m in moves):
                        return True
        return False

    def get_all_moves(self, player):
        all_moves = []
        has_jumps = self.has_jumps_available(player)
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves(row, col)
                    if has_jumps:
                        moves = [m for m in moves if m['captured']]
                    all_moves.extend(moves)
        return all_moves

    def get_all_moves_for_board(self, board, player):
        all_moves = []
        has_jumps = False
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves_for_board(board, row, col)
                    if any(m['captured'] for m in moves):
                        has_jumps = True
                        break
            if has_jumps:
                break
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves_for_board(board, row, col)
                    if has_jumps:
                        moves = [m for m in moves if m['captured']]
                    all_moves.extend(moves)
        return all_moves

    def get_possible_moves_for_board(self, board, row, col):
        piece = board[row][col]
        moves = []
        if piece == 0:
            return moves
        is_king = abs(piece) == 2
        player = 1 if piece > 0 else -1
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else (
            [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]
        )
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == 0:
                moves.append({'start': (row, col), 'end': (new_row, new_col), 'captured': None})
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if (0 <= jump_row < 8 and 0 <= jump_col < 8 and 
                board[new_row][new_col] * player < 0 and
                board[jump_row][jump_col] == 0):
                moves.append({'start': (row, col), 'end': (jump_row, jump_col), 'captured': (new_row, new_col)})
        return moves

    def evaluate_board_static(self, board):
        score = 0
        men_value = 30
        king_value = 90
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == 0:
                    continue
                side = -1 if piece < 0 else 1
                is_man = (abs(piece) == 1)
                base = king_value if not is_man else men_value
                advance = (row if side == -1 else (7 - row)) if is_man else 0
                central = 2 if (2 <= row <= 5 and 2 <= col <= 5) else 0
                edge_pen = 1 if (row in (0, 7) or col in (0, 7)) else 0
                contrib = base + 0.6 * advance + central - edge_pen
                score += (-side) * contrib
        try:
            black_moves = len(self.get_all_moves_for_board(board, -1))
            red_moves   = len(self.get_all_moves_for_board(board,  1))
            score += 0.5 * (black_moves - red_moves)
        except Exception:
            pass
        return score

    def material_balance(self, board):
        """
        > 0: Vorteil für Schwarz (-1), < 0: Vorteil für Rot (+1).
        (kleine Werte, nur für Draw-Bewertung)
        """
        score = 0
        for r in range(8):
            for c in range(8):
                p = board[r][c]
                if p == 0:
                    continue
                val = 3 if abs(p) == 2 else 1  # König schwerer gewichten
                if p < 0:   # Schwarz
                    score += val
                else:       # Rot
                    score -= val
        return score

    def make_move(self, move, is_ai=False):
        start_row, start_col = move['start']
        end_row, end_col = move['end']
        captured = move['captured']
        piece = self.board[start_row][start_col]

        progress = False  # Schlag/Beförderung passiert?
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = 0

        if captured:
            cap_row, cap_col = captured
            self.board[cap_row][cap_col] = 0
            self.game_stats.captured_pieces[self.current_player] += 1
            progress = True

        # Beförderung
        if piece == 1 and end_row == 0:
            self.board[end_row][end_col] = 2
            progress = True
            if self.game_stats.first_king_time[1] is None:
                self.game_stats.first_king_time[1] = time.time() - self.game_stats.game_start_time
        elif piece == -1 and end_row == 7:
            self.board[end_row][end_col] = -2
            progress = True
            if self.game_stats.first_king_time[-1] is None:
                self.game_stats.first_king_time[-1] = time.time() - self.game_stats.game_start_time

        # Mehrfachsprung
        if captured:
            further_jumps = self.get_possible_moves(end_row, end_col)
            jump_moves = [m for m in further_jumps if m['captured']]
            if jump_moves:
                if is_ai:
                    best_jump = jump_moves[0]
                    if self.gui:
                        self.root.after(min(300, self.speed_var.get()), 
                                        lambda: self.make_move(best_jump, is_ai=True))
                    else:
                        self.make_move(best_jump, is_ai=True)
                    if self.gui:
                        self.draw_pieces()
                    return
                else:
                    pass  # Human-Variante entfällt

        # Halbzugzähler aktualisieren
        self.halfmove_clock = 0 if progress else self.halfmove_clock + 1

        # Zugwechsel und Buchhaltung
        self.must_jump = False
        self.jump_piece = None
        self.selected_pos = None
        self.selected_piece = None
        self.current_player *= -1
        self.game_stats.moves_count += 1

        # Stellung nach dem Zug für den neuen Spieler zählen
        self.record_position()

        if self.gui:
            self.draw_pieces()
            self.update_display()

        # Ende prüfen
        if self.check_game_over():
            return

        # Nächster Zug (nur AIvAI)
        if self.gui and not self.pause_var.get():
            self.root.after(self.speed_var.get(), self.ai_move)
        elif not self.gui:
            self.ai_move()

    def ai_move(self):
        if self.game_over:
            return
        current_ai = self.red_ai if self.current_player == 1 else self.black_ai
        if current_ai:
            move = current_ai.get_move(self)
            if move:
                self.make_move(move, is_ai=True)

    def check_game_over(self):
        # Siegbedingungen
        red_pieces = sum(1 for row in self.board for piece in row if piece > 0)
        black_pieces = sum(1 for row in self.board for piece in row if piece < 0)
        winner = None
        if red_pieces == 0:
            winner = -1
        elif black_pieces == 0:
            winner = 1
        elif not self.get_all_moves(self.current_player):
            winner = -self.current_player

        # Remisbedingungen
        draw_reason = None
        if winner is None and self.is_draw_by_50_move_rule():
            draw_reason = "Draw by 50-move rule"
        if winner is None and self.is_draw_by_repetition():
            draw_reason = "Draw by threefold repetition"

        if winner or draw_reason:
            self.game_over = True
            if self.gui:
                if draw_reason:
                    messagebox.showinfo("Game Over", draw_reason)
                else:
                    winner_text = "Red" if winner == 1 else "Black"
                    winner_ai = self.red_ai if winner == 1 else self.black_ai
                    if winner_ai:
                        winner_text += f" ({winner_ai.name})"
                    messagebox.showinfo("Game Over", f"{winner_text} wins!")
            return True
        return False

    def run(self):
        if self.gui:
            self.root.after(1000, self.ai_move)
            self.root.mainloop()
        else:
            while not self.game_over:
                self.ai_move()

def main():
    root = tk.Tk()
    root.title("Checkers AI — AI vs AI")
    root.geometry("500x350")

    tk.Label(root, text="Select AIs", font=('Arial', 16, 'bold')).pack(pady=10)

    red_frame = tk.LabelFrame(root, text="Red AI", font=('Arial', 12, 'bold'))
    red_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)
    red_var = tk.StringVar(value="GREEDY")
    for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
        tk.Radiobutton(red_frame, text=ai_type.value, variable=red_var, value=ai_type.name, font=('Arial', 10)).pack(pady=5, anchor="w")

    black_frame = tk.LabelFrame(root, text="Black AI", font=('Arial', 12, 'bold'))
    black_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)
    black_var = tk.StringVar(value="MINIMAX_SMART")
    for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
        tk.Radiobutton(black_frame, text=ai_type.value, variable=black_var, value=ai_type.name, font=('Arial', 10)).pack(pady=5, anchor="w")

    def launch():
        red_type = PlayerType[red_var.get()]
        black_type = PlayerType[black_var.get()]
        root.destroy()
        game = CheckersGame(mode="AIvAI", red_ai_type=red_type, black_ai_type=black_type)
        game.run()

    tk.Button(root, text="Start Game", command=launch, bg='green', fg='white', font=('Arial', 12)).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main()
