import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import copy
import math
import random
import time
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime

class PlayerType(Enum):
    HUMAN = "Human"
    GREEDY = "Greedy Bot"
    MINIMAX_SIMPLE = "Simple Minimax"
    MINIMAX_SMART = "Smart Minimax"
    RANDOM = "Random Bot"

class GameStats:
    """Klasse für Spielstatistiken"""
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
        
class TournamentStats:
    """Klasse für Turnierstatistiken"""
    def __init__(self):
        self.games_played = 0
        self.wins = {}
        self.total_moves = []
        self.total_game_times = []
        self.avg_positions_evaluated = {}
        self.matchup_results = {}  # (player1, player2) -> [wins_p1, wins_p2]

class BaseAI(ABC):
    """Abstrakte Basisklasse für alle KI-Varianten"""
    def __init__(self, player, name):
        self.player = player
        self.name = name
        self.stats = GameStats()
    
    @abstractmethod
    def get_move(self, game):
        """Muss von jeder KI implementiert werden"""
        pass
    
    def reset_stats(self):
        self.stats.reset()

class RandomAI(BaseAI):
    """Zufällige KI - macht zufällige legale Züge"""
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
    """Greedy KI - schaut nur 1 Zug voraus"""
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
            # Simuliere den Zug
            new_board = game.simulate_complete_move(game.board, move)
            score = self.evaluate_board(new_board, game)
            self.stats.positions_evaluated += 1
            
            # Bevorzuge Schlagzüge
            if move['captured']:
                score += 15
            
            if score > best_score:
                best_score = score
                best_move = move
        
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    
    def evaluate_board(self, board, game):
        """Einfache Bewertungsfunktion"""
        score = 0
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece != 0:
                    piece_value = 10 if abs(piece) == 1 else 50
                    if piece * self.player > 0:
                        score += piece_value
                    else:
                        score -= piece_value
        return score

class SimpleMinimaxAI(BaseAI):
    """Minimax ohne Alpha-Beta-Pruning"""
    def __init__(self, player, depth=4):
        super().__init__(player, f"Simple Minimax (Depth {depth}, No Pruning)")
        self.depth = depth
    
    def get_move(self, game):
        start_time = time.time()
        _, best_move = self.minimax(game, game.board, self.depth, self.player == -1)
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    
    def minimax(self, game, board, depth, maximizing_player):
        """Minimax OHNE Alpha-Beta-Pruning"""
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
                new_board = game.simulate_complete_move(board, move)
                eval_score, _ = self.minimax(game, new_board, depth - 1, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in possible_moves:
                new_board = game.simulate_complete_move(board, move)
                eval_score, _ = self.minimax(game, new_board, depth - 1, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
            
            return min_eval, best_move

class SmartMinimaxAI(BaseAI):
    """Minimax MIT Alpha-Beta-Pruning"""
    def __init__(self, player, depth=5):
        super().__init__(player, f"Smart Minimax (Depth {depth}, Alpha-Beta)")
        self.depth = depth
    
    def get_move(self, game):
        start_time = time.time()
        _, best_move = self.minimax(game, game.board, self.depth, -math.inf, math.inf, self.player == -1)
        self.stats.move_times.append(time.time() - start_time)
        return best_move
    
    def minimax(self, game, board, depth, alpha, beta, maximizing_player):
        """Minimax MIT Alpha-Beta-Pruning"""
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
                new_board = game.simulate_complete_move(board, move)
                eval_score, _ = self.minimax(game, new_board, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    self.stats.pruning_cuts += 1
                    break  # Alpha-Beta Pruning
            
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in possible_moves:
                new_board = game.simulate_complete_move(board, move)
                eval_score, _ = self.minimax(game, new_board, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    self.stats.pruning_cuts += 1
                    break  # Alpha-Beta Pruning
            
            return min_eval, best_move

class CheckersGame:
    def __init__(self, mode="PvAI", red_ai_type=None, black_ai_type=None, gui=True, tournament_callback=None):
        self.board = self.create_initial_board()
        self.current_player = 1
        self.selected_piece = None
        self.selected_pos = None
        self.must_jump = False
        self.jump_piece = None
        self.mode = mode
        self.gui = gui
        self.tournament_callback = tournament_callback
        self.game_over = False
        
        # Statistiken
        self.game_stats = GameStats()
        
        # KI-Spieler erstellen
        self.red_ai = self.create_ai(1, red_ai_type) if red_ai_type else None
        self.black_ai = self.create_ai(-1, black_ai_type) if black_ai_type else None
        
        if gui:
            self.setup_gui()
    
    def create_ai(self, player, ai_type):
        """Erstellt eine KI basierend auf dem Typ"""
        if ai_type == PlayerType.RANDOM:
            return RandomAI(player)
        elif ai_type == PlayerType.GREEDY:
            return GreedyAI(player)
        elif ai_type == PlayerType.MINIMAX_SIMPLE:
            return SimpleMinimaxAI(player, depth=4)
        elif ai_type == PlayerType.MINIMAX_SMART:
            return SmartMinimaxAI(player, depth=5)
        return None
    
    def setup_gui(self):
        """GUI Setup"""
        self.root = tk.Tk()
        self.root.title("Advanced Checkers AI Tournament System")
        self.root.resizable(False, False)
        
        # Hauptframe
        main_frame = tk.Frame(self.root)
        main_frame.pack()
        
        # Linke Seite - Spielbrett
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(left_frame, width=640, height=640, bg='white')
        self.canvas.pack()
        
        if self.mode == "PvAI":
            self.canvas.bind("<Button-1>", self.on_click)
        
        # Info Label
        self.info_label = tk.Label(left_frame, text="", font=('Arial', 14))
        self.info_label.pack(pady=5)
        
        # Rechte Seite - Statistiken
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
        
        # Spieler Info
        player_frame = tk.LabelFrame(right_frame, text="Current Game", font=('Arial', 12, 'bold'))
        player_frame.pack(fill=tk.X, pady=5)
        
        self.red_player_label = tk.Label(player_frame, text="", fg="red")
        self.red_player_label.pack()
        
        self.black_player_label = tk.Label(player_frame, text="", fg="black")
        self.black_player_label.pack()
        
        # Live Stats
        stats_frame = tk.LabelFrame(right_frame, text="Live Statistics", font=('Arial', 12, 'bold'))
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.stats_text = tk.Text(stats_frame, width=40, height=15, font=('Courier', 9))
        self.stats_text.pack()
        
        # Kontroll-Buttons
        control_frame = tk.Frame(right_frame)
        control_frame.pack(pady=10)
        
        self.speed_var = tk.IntVar(value=500)
        tk.Label(control_frame, text="AI Speed (ms):").pack()
        tk.Scale(control_frame, from_=50, to=2000, orient=tk.HORIZONTAL, 
                variable=self.speed_var, length=200).pack()
        
        if self.mode == "AIvAI":
            self.pause_var = tk.BooleanVar(value=False)
            tk.Checkbutton(control_frame, text="Pause", variable=self.pause_var).pack()
        
        self.draw_board()
        self.draw_pieces()
        self.update_display()
    
    def create_initial_board(self):
        """Erstellt das initiale Spielbrett"""
        board = [[0 for _ in range(8)] for _ in range(8)]
        
        # Schwarze Steine - oben
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = -1
        
        # Rote Steine - unten
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 1
                    
        return board
    
    def draw_board(self):
        """Zeichnet das Schachbrett"""
        if not self.gui:
            return
        self.canvas.delete("board")
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 80, row * 80
                x2, y2 = x1 + 80, y1 + 80
                
                if (row + col) % 2 == 0:
                    color = '#F0D9B5'
                else:
                    color = '#B58863'
                    
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, tags="board")
    
    def draw_pieces(self):
        """Zeichnet alle Spielsteine"""
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
        
        if self.selected_pos:
            moves = self.get_possible_moves(self.selected_pos[0], self.selected_pos[1])
            for move in moves:
                end_row, end_col = move['end']
                x, y = end_col * 80 + 40, end_row * 80 + 40
                self.canvas.create_oval(x-10, y-10, x+10, y+10, 
                                      fill='lime', outline='green', width=2, tags="highlight")
    
    def update_display(self):
        """Aktualisiert die Anzeige"""
        if not self.gui:
            return
            
        # Spieler Info
        if self.current_player == 1:
            player_text = "Red's Turn"
            current_ai = self.red_ai
        else:
            player_text = "Black's Turn"
            current_ai = self.black_ai
        
        if current_ai:
            player_text += f" ({current_ai.name})"
        
        self.info_label.config(text=player_text)
        
        # Update player labels
        red_text = "Red: " + (self.red_ai.name if self.red_ai else "Human")
        black_text = "Black: " + (self.black_ai.name if self.black_ai else "Human")
        self.red_player_label.config(text=red_text)
        self.black_player_label.config(text=black_text)
        
        # Statistiken
        self.update_stats_display()
    
    def update_stats_display(self):
        """Aktualisiert die Statistik-Anzeige"""
        if not self.gui:
            return
            
        stats_text = []
        
        # Spielstatistiken
        stats_text.append(f"Move: {self.game_stats.moves_count}")
        
        # Steine zählen
        red_pieces = sum(1 for row in self.board for piece in row if piece > 0)
        black_pieces = sum(1 for row in self.board for piece in row if piece < 0)
        red_kings = sum(1 for row in self.board for piece in row if piece == 2)
        black_kings = sum(1 for row in self.board for piece in row if piece == -2)
        
        stats_text.append(f"\nPieces:")
        stats_text.append(f"Red: {red_pieces} ({red_kings} kings)")
        stats_text.append(f"Black: {black_pieces} ({black_kings} kings)")
        
        # KI Statistiken
        if self.red_ai and self.red_ai.stats.positions_evaluated > 0:
            stats_text.append(f"\nRed AI Stats:")
            stats_text.append(f"Positions: {self.red_ai.stats.positions_evaluated}")
            if hasattr(self.red_ai, 'stats') and self.red_ai.stats.pruning_cuts > 0:
                stats_text.append(f"Pruning Cuts: {self.red_ai.stats.pruning_cuts}")
            if self.red_ai.stats.move_times:
                avg_time = sum(self.red_ai.stats.move_times) / len(self.red_ai.stats.move_times)
                stats_text.append(f"Avg Move Time: {avg_time:.3f}s")
        
        if self.black_ai and self.black_ai.stats.positions_evaluated > 0:
            stats_text.append(f"\nBlack AI Stats:")
            stats_text.append(f"Positions: {self.black_ai.stats.positions_evaluated}")
            if hasattr(self.black_ai, 'stats') and self.black_ai.stats.pruning_cuts > 0:
                stats_text.append(f"Pruning Cuts: {self.black_ai.stats.pruning_cuts}")
            if self.black_ai.stats.move_times:
                avg_time = sum(self.black_ai.stats.move_times) / len(self.black_ai.stats.move_times)
                stats_text.append(f"Avg Move Time: {avg_time:.3f}s")
        
        # Effizienz-Vergleich
        if (self.red_ai and self.black_ai and 
            self.red_ai.stats.positions_evaluated > 0 and 
            self.black_ai.stats.positions_evaluated > 0):
            
            ratio = self.black_ai.stats.positions_evaluated / self.red_ai.stats.positions_evaluated
            stats_text.append(f"\nEfficiency Ratio: {ratio:.2f}")
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, '\n'.join(stats_text))
    
    def on_click(self, event):
        """Behandelt Mausklicks"""
        if self.current_player != 1 or self.red_ai:
            return
            
        col = event.x // 80
        row = event.y // 80
        
        if 0 <= row < 8 and 0 <= col < 8:
            self.handle_click(row, col)
    
    def handle_click(self, row, col):
        """Behandelt Klicks auf das Spielbrett"""
        piece = self.board[row][col]
        
        if self.must_jump and self.jump_piece:
            if (row, col) == self.jump_piece:
                self.selected_pos = (row, col)
                self.selected_piece = piece
                self.draw_pieces()
                return
            elif self.selected_pos:
                moves = self.get_possible_moves(self.selected_pos[0], self.selected_pos[1])
                jump_moves = [m for m in moves if m['captured']]
                
                for move in jump_moves:
                    if move['end'] == (row, col):
                        self.make_move(move)
                        return
                return
        
        if piece * self.current_player > 0:
            self.selected_pos = (row, col)
            self.selected_piece = piece
            self.draw_pieces()
        
        elif self.selected_pos:
            moves = self.get_possible_moves(self.selected_pos[0], self.selected_pos[1])
            for move in moves:
                if move['end'] == (row, col):
                    self.make_move(move)
                    return
            
            self.selected_pos = None
            self.selected_piece = None
            self.draw_pieces()
    
    def get_possible_moves(self, row, col):
        """Gibt alle möglichen Züge für einen Stein zurück"""
        piece = self.board[row][col]
        moves = []
        
        if piece == 0:
            return moves
        
        is_king = abs(piece) == 2
        player = 1 if piece > 0 else -1
        
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if player == 1:
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and self.board[new_row][new_col] == 0:
                moves.append({
                    'start': (row, col),
                    'end': (new_row, new_col),
                    'captured': None
                })
            
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if (0 <= jump_row < 8 and 0 <= jump_col < 8 and 
                self.board[new_row][new_col] * player < 0 and
                self.board[jump_row][jump_col] == 0):
                
                moves.append({
                    'start': (row, col),
                    'end': (jump_row, jump_col),
                    'captured': (new_row, new_col)
                })
        
        return moves
    
    def has_jumps_available(self, player):
        """Prüft ob Sprünge verfügbar sind"""
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves(row, col)
                    if any(move['captured'] for move in moves):
                        return True
        return False
    
    def get_all_moves(self, player):
        """Gibt alle möglichen Züge für einen Spieler zurück"""
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
        """Hilfsfunktion für Minimax - alle Züge für ein Brett"""
        all_moves = []
        has_jumps = False
        
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece * player > 0:
                    moves = self.get_possible_moves_for_board(board, row, col)
                    if any(move['captured'] for move in moves):
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
        """Hilfsfunktion für Minimax"""
        piece = board[row][col]
        moves = []
        
        if piece == 0:
            return moves
        
        is_king = abs(piece) == 2
        player = 1 if piece > 0 else -1
        
        if is_king:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            if player == 1:
                directions = [(-1, -1), (-1, 1)]
            else:
                directions = [(1, -1), (1, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8 and board[new_row][new_col] == 0:
                moves.append({
                    'start': (row, col),
                    'end': (new_row, new_col),
                    'captured': None
                })
            
            jump_row, jump_col = row + 2*dr, col + 2*dc
            if (0 <= jump_row < 8 and 0 <= jump_col < 8 and 
                board[new_row][new_col] * player < 0 and
                board[jump_row][jump_col] == 0):
                
                moves.append({
                    'start': (row, col),
                    'end': (jump_row, jump_col),
                    'captured': (new_row, new_col)
                })
        
        return moves
    
    def simulate_complete_move(self, board, move):
        """Simuliert einen kompletten Zug inkl. aller Mehrfachsprünge"""
        new_board = copy.deepcopy(board)
        
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
            
            if piece == 1 and end_row == 0:
                new_board[end_row][end_col] = 2
            elif piece == -1 and end_row == 7:
                new_board[end_row][end_col] = -2
            
            if captured:
                further_jumps = self.get_possible_moves_for_board(new_board, end_row, end_col)
                jump_moves = [m for m in further_jumps if m['captured']]
                
                if jump_moves:
                    move = jump_moves[0]
                    continue
            
            break
        
        return new_board
    
        """Statische Bewertungsfunktion für das Brett"""
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                
                if piece != 0:
                    piece_value = 10 if abs(piece) == 1 else 50
                    
                    if piece > 0:
                        score -= piece_value
                        score -= (7 - row) * 2
                    else:
                        score += piece_value
                        score += row * 2
                    
                    if 2 <= row <= 5 and 2 <= col <= 5:
                        score += 5 if piece < 0 else -5
                    
                    if row == 0 or row == 7 or col == 0 or col == 7:
                        score += 3 if piece < 0 else -3
        
        return score
    
    def evaluate_board_static(self, board):
        """
        Positiv = gut für Schwarz (-1), negativ = gut für Rot (+1).
        Material zählt am meisten; leichte Boni für Fortschritt (nur Men), Zentrum, Mobilität.
        """
        score = 0
        men_value = 30
        king_value = 90

        # Material, Fortschritt, Position
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece == 0:
                    continue

                side = -1 if piece < 0 else 1  # Schwarz=-1, Rot=+1
                is_man = (abs(piece) == 1)

                base = king_value if not is_man else men_value

                # Fortschritt nur für "Men"
                if is_man:
                    advance = row if side == -1 else (7 - row)  # Richtung Zielreihe
                else:
                    advance = 0

                # Zentrum leicht belohnen
                central = 2 if (2 <= row <= 5 and 2 <= col <= 5) else 0
                # Rand leicht bestrafen
                edge_pen = 1 if (row in (0, 7) or col in (0, 7)) else 0

                contrib = base + 0.6 * advance + central - edge_pen

                # Positiv für Schwarz, negativ für Rot (beibehaltene Konvention)
                score += (-side) * contrib

        # Mobilität: mehr legale Züge ist leicht besser
        try:
            black_moves = len(self.get_all_moves_for_board(board, -1))
            red_moves   = len(self.get_all_moves_for_board(board,  1))
            score += 0.5 * (black_moves - red_moves)
        except Exception:
            pass

        return score


    def make_move(self, move, is_ai=False):
        """Führt einen Zug aus"""
        start_row, start_col = move['start']
        end_row, end_col = move['end']
        captured = move['captured']
        
        piece = self.board[start_row][start_col]
        
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = 0
        
        if captured:
            cap_row, cap_col = captured
            self.board[cap_row][cap_col] = 0
            self.game_stats.captured_pieces[self.current_player] += 1
        
        if piece == 1 and end_row == 0:
            self.board[end_row][end_col] = 2
            if self.game_stats.first_king_time[1] is None:
                self.game_stats.first_king_time[1] = time.time() - self.game_stats.game_start_time
        elif piece == -1 and end_row == 7:
            self.board[end_row][end_col] = -2
            if self.game_stats.first_king_time[-1] is None:
                self.game_stats.first_king_time[-1] = time.time() - self.game_stats.game_start_time
        
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
                    self.must_jump = True
                    self.jump_piece = (end_row, end_col)
                    self.selected_pos = (end_row, end_col)
                    if self.gui:
                        self.draw_pieces()
                    return
        
        self.must_jump = False
        self.jump_piece = None
        self.selected_pos = None
        self.selected_piece = None
        
        self.current_player *= -1
        self.game_stats.moves_count += 1
        
        if self.gui:
            self.draw_pieces()
            self.update_display()
        
        if self.check_game_over():
            return
        
        # --- NEU: 50-Züge-Regel zählen ---
        if captured:
            self.moves_without_capture = 0
        else:
            self.moves_without_capture += 1

        # Nächster Zug
        if self.mode == "AIvAI":
            if self.gui and not self.pause_var.get():
                self.root.after(self.speed_var.get(), self.ai_move)
            elif not self.gui:
                self.ai_move()
        elif self.mode == "PvAI":
            if (self.current_player == 1 and self.red_ai) or (self.current_player == -1 and self.black_ai):
                if self.gui:
                    self.root.after(self.speed_var.get(), self.ai_move)
                else:
                    self.ai_move()
    
    def ai_move(self):
        """KI macht einen Zug"""
        if self.game_over:
            return
            
        current_ai = self.red_ai if self.current_player == 1 else self.black_ai
        
        if current_ai:
            move = current_ai.get_move(self)
            if move:
                self.make_move(move, is_ai=True)
    
    def check_game_over(self):
        """Prüft ob das Spiel beendet ist"""
        red_pieces = sum(1 for row in self.board for piece in row if piece > 0)
        black_pieces = sum(1 for row in self.board for piece in row if piece < 0)
        
        winner = None
        
        if red_pieces == 0:
            winner = -1
        elif black_pieces == 0:
            winner = 1
        elif not self.get_all_moves(self.current_player):
            winner = -self.current_player
        
        if winner:
            self.game_over = True
            if self.tournament_callback:
                self.tournament_callback(winner)
            elif self.gui:
                winner_text = "Red" if winner == 1 else "Black"
                winner_ai = self.red_ai if winner == 1 else self.black_ai
                if winner_ai:
                    winner_text += f" ({winner_ai.name})"
                messagebox.showinfo("Game Over", f"{winner_text} wins!")
            return True
        
        return False
    
    def run(self):
        """Startet das Spiel"""
        if self.gui:
            # Starte das erste KI-Spiel wenn AIvAI
            if self.mode == "AIvAI":
                self.root.after(1000, self.ai_move)
            elif self.mode == "PvAI" and self.current_player == -1 and self.black_ai:
                self.root.after(1000, self.ai_move)
            
            self.root.mainloop()
        else:
            # Spiele ohne GUI
            while not self.game_over:
                self.ai_move()

class TournamentManager:
    """Verwaltet Turniere zwischen verschiedenen KIs"""
    def __init__(self):
        self.stats = TournamentStats()
        self.current_game = None
        self.tournament_window = None
        
    def run_tournament(self, player_types, games_per_matchup=10):
        """Führt ein komplettes Turnier durch"""
        self.setup_tournament_window()
        
        # Initialisiere Statistiken
        for ptype in player_types:
            self.stats.wins[ptype] = 0
        
        matchups = []
        for i, p1 in enumerate(player_types):
            for p2 in player_types[i+1:]:
                matchups.append((p1, p2))
        
        total_games = len(matchups) * games_per_matchup * 2  # Jeder spielt als rot und schwarz
        games_played = 0
        
        for p1, p2 in matchups:
            # Spiele mit p1 als rot, p2 als schwarz
            for _ in range(games_per_matchup):
                self.update_tournament_display(f"Game {games_played+1}/{total_games}: {p1.value} (Red) vs {p2.value} (Black)")
                winner = self.play_single_game(p1, p2)
                games_played += 1
                
            # Spiele mit p2 als rot, p1 als schwarz
            for _ in range(games_per_matchup):
                self.update_tournament_display(f"Game {games_played+1}/{total_games}: {p2.value} (Red) vs {p1.value} (Black)")
                winner = self.play_single_game(p2, p1)
                games_played += 1
        
        self.show_final_results()
    
    def play_single_game(self, red_type, black_type):
        """Spielt ein einzelnes Spiel"""
        def game_callback(winner):
            if winner == 1:
                self.stats.wins[red_type] = self.stats.wins.get(red_type, 0) + 1
                winner_type = red_type
            else:
                self.stats.wins[black_type] = self.stats.wins.get(black_type, 0) + 1
                winner_type = black_type
            
            matchup = (red_type, black_type)
            if matchup not in self.stats.matchup_results:
                self.stats.matchup_results[matchup] = [0, 0]
            
            if winner == 1:
                self.stats.matchup_results[matchup][0] += 1
            else:
                self.stats.matchup_results[matchup][1] += 1
            
            self.stats.games_played += 1
            self.update_stats_display()
        
        game = CheckersGame(mode="AIvAI", red_ai_type=red_type, 
                           black_ai_type=black_type, gui=False, 
                           tournament_callback=game_callback)
        game.run()
        
        # Sammle Statistiken
        self.stats.total_moves.append(game.game_stats.moves_count)
        
        return game
    
    def setup_tournament_window(self):
        """Erstellt das Turnierfenster"""
        self.tournament_window = tk.Toplevel()
        self.tournament_window.title("Tournament Progress")
        self.tournament_window.geometry("600x500")
        
        # Titel
        title_label = tk.Label(self.tournament_window, 
                              text="AI Tournament", 
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # Status
        self.status_label = tk.Label(self.tournament_window, 
                                    text="Starting tournament...", 
                                    font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        # Statistik-Bereich
        stats_frame = tk.LabelFrame(self.tournament_window, 
                                   text="Current Standings", 
                                   font=('Arial', 12, 'bold'))
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.stats_display = scrolledtext.ScrolledText(stats_frame, 
                                                      width=70, height=20, 
                                                      font=('Courier', 10))
        self.stats_display.pack(padx=10, pady=10)
    
    def update_tournament_display(self, status):
        """Aktualisiert die Turnieranzeige"""
        if self.tournament_window:
            self.status_label.config(text=status)
            self.tournament_window.update()
    
    def update_stats_display(self):
        """Aktualisiert die Statistikanzeige"""
        if not self.tournament_window:
            return
            
        display_text = []
        
        # Gesamtstatistiken
        display_text.append(f"Games Played: {self.stats.games_played}\n")
        display_text.append("="*50 + "\n")
        
        # Gewinnstatistiken
        display_text.append("Win Statistics:\n")
        display_text.append("-"*30 + "\n")
        
        sorted_wins = sorted(self.stats.wins.items(), key=lambda x: x[1], reverse=True)
        for player_type, wins in sorted_wins:
            win_rate = (wins / max(self.stats.games_played, 1)) * 100
            display_text.append(f"{player_type.value:20} | {wins:3} wins | {win_rate:5.1f}%\n")
        
        # Matchup-Details
        if self.stats.matchup_results:
            display_text.append("\n" + "="*50 + "\n")
            display_text.append("Head-to-Head Results:\n")
            display_text.append("-"*30 + "\n")
            
            for (p1, p2), (wins_p1, wins_p2) in self.stats.matchup_results.items():
                total = wins_p1 + wins_p2
                if total > 0:
                    p1_rate = (wins_p1 / total) * 100
                    display_text.append(f"{p1.value:15} vs {p2.value:15}\n")
                    display_text.append(f"  → {wins_p1}-{wins_p2} ({p1_rate:.0f}%)\n")
        
        # Durchschnittliche Spiellänge
        if self.stats.total_moves:
            avg_moves = sum(self.stats.total_moves) / len(self.stats.total_moves)
            display_text.append(f"\nAverage Game Length: {avg_moves:.1f} moves\n")
        
        self.stats_display.delete(1.0, tk.END)
        self.stats_display.insert(1.0, ''.join(display_text))
        self.tournament_window.update()
    
    def show_final_results(self):
        """Zeigt die finalen Turnierergebnisse"""
        self.update_tournament_display("Tournament Complete!")
        
        # Füge finale Zusammenfassung hinzu
        summary = "\n" + "="*50 + "\n"
        summary += "FINAL TOURNAMENT RESULTS\n"
        summary += "="*50 + "\n\n"
        
        # Rangliste
        sorted_wins = sorted(self.stats.wins.items(), key=lambda x: x[1], reverse=True)
        for i, (player_type, wins) in enumerate(sorted_wins, 1):
            games_without_draws = self.stats.games_played - self.stats.draws
            win_rate = (wins / games_without_draws * 100) if games_without_draws > 0 else 0
            summary += f"{i}. {player_type.value}: {wins} wins ({win_rate:.1f}%)\n"
        
        summary += f"\nTotal Draws: {self.stats.draws} ({self.stats.draws/self.stats.games_played*100:.1f}%)\n"
        
        self.stats_display.insert(tk.END, summary)
        
        # Speichere Ergebnisse
        self.save_results()
    
    def save_results(self):
        """Speichert die Turnierergebnisse in einer Datei"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tournament_results_{timestamp}.json"
        
        results = {
            "games_played": self.stats.games_played,
            "wins": {k.value: v for k, v in self.stats.wins.items()},
            "matchup_results": {f"{k[0].value}_vs_{k[1].value}": v 
                               for k, v in self.stats.matchup_results.items()},
            "average_game_length": sum(self.stats.total_moves) / len(self.stats.total_moves) if self.stats.total_moves else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.tournament_window:
            self.stats_display.insert(tk.END, f"\n\nResults saved to {filename}")

def main():
    """Hauptfunktion mit Menü"""
    root = tk.Tk()
    root.title("Advanced Checkers AI System")
    root.geometry("500x600")
    
    # Titel
    title = tk.Label(root, text="Advanced Checkers AI", font=('Arial', 20, 'bold'))
    title.pack(pady=20)
    
    # Beschreibung
    desc = tk.Label(root, text="Choose a game mode:", font=('Arial', 12))
    desc.pack(pady=10)
    
    # Spielmodus-Auswahl
    mode_frame = tk.LabelFrame(root, text="Game Modes", font=('Arial', 12, 'bold'))
    mode_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
    
    def start_pvp():
        root.destroy()
        game = CheckersGame(mode="PvP")
        game.run()
    
    def start_pvai():
        def launch():
            ai_type = PlayerType[ai_var.get()]
            root.destroy()
            game = CheckersGame(mode="PvAI", black_ai_type=ai_type)
            game.run()
        
        # KI-Auswahl
        ai_frame = tk.Toplevel(root)
        ai_frame.title("Select AI Opponent")
        ai_frame.geometry("300x250")
        
        tk.Label(ai_frame, text="Choose AI Type:", font=('Arial', 12)).pack(pady=10)
        
        ai_var = tk.StringVar(value="GREEDY")
        
        for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, 
                        PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
            tk.Radiobutton(ai_frame, text=ai_type.value, 
                          variable=ai_var, value=ai_type.name,
                          font=('Arial', 10)).pack(pady=5)
        
        tk.Button(ai_frame, text="Start Game", command=launch, 
                 bg='green', fg='white', font=('Arial', 12)).pack(pady=20)
    
    def start_aivai():
        def launch():
            red_type = PlayerType[red_var.get()]
            black_type = PlayerType[black_var.get()]
            root.destroy()
            game = CheckersGame(mode="AIvAI", red_ai_type=red_type, black_ai_type=black_type)
            game.run()
        
        # KI-Auswahl
        ai_frame = tk.Toplevel(root)
        ai_frame.title("Select AIs")
        ai_frame.geometry("500x350")
        
        # Red AI
        red_frame = tk.LabelFrame(ai_frame, text="Red AI", font=('Arial', 12, 'bold'))
        red_frame.pack(side=tk.LEFT, padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        red_var = tk.StringVar(value="GREEDY")
        for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, 
                        PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
            tk.Radiobutton(red_frame, text=ai_type.value, 
                          variable=red_var, value=ai_type.name,
                          font=('Arial', 9)).pack(pady=5)
        
        # Black AI
        black_frame = tk.LabelFrame(ai_frame, text="Black AI", font=('Arial', 12, 'bold'))
        black_frame.pack(side=tk.RIGHT, padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        black_var = tk.StringVar(value="MINIMAX_SMART")
        for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, 
                        PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
            tk.Radiobutton(black_frame, text=ai_type.value, 
                          variable=black_var, value=ai_type.name,
                          font=('Arial', 9)).pack(pady=5)
        
        tk.Button(ai_frame, text="Start Game", command=launch, 
                 bg='green', fg='white', font=('Arial', 12)).pack(pady=20)
    
    def start_tournament():
        def launch():
            selected = []
            for var, ptype in ai_vars:
                if var.get():
                    selected.append(ptype)
            
            if len(selected) < 2:
                messagebox.showwarning("Selection Error", "Please select at least 2 AIs!")
                return
            
            games = int(games_var.get())
            tourney_frame.destroy()
            
            manager = TournamentManager()
            manager.run_tournament(selected, games_per_matchup=games)
        
        # Turnier-Setup
        tourney_frame = tk.Toplevel(root)
        tourney_frame.title("Tournament Setup")
        tourney_frame.geometry("400x450")
        
        tk.Label(tourney_frame, text="Select AIs for Tournament:", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        
        ai_vars = []
        for ai_type in [PlayerType.RANDOM, PlayerType.GREEDY, 
                        PlayerType.MINIMAX_SIMPLE, PlayerType.MINIMAX_SMART]:
            var = tk.BooleanVar(value=True if ai_type != PlayerType.RANDOM else False)
            ai_vars.append((var, ai_type))
            tk.Checkbutton(tourney_frame, text=ai_type.value, 
                          variable=var, font=('Arial', 10)).pack(pady=5)
        
        tk.Label(tourney_frame, text="\nGames per matchup:", 
                font=('Arial', 10)).pack(pady=10)
        
        games_var = tk.StringVar(value="10")
        tk.Spinbox(tourney_frame, from_=1, to=100, textvariable=games_var, 
                  width=10, font=('Arial', 10)).pack()
        
        tk.Button(tourney_frame, text="Start Tournament", command=launch, 
                 bg='red', fg='white', font=('Arial', 12, 'bold')).pack(pady=30)
    
    # Buttons
    btn_frame = tk.Frame(mode_frame)
    btn_frame.pack(expand=True)
    
    tk.Button(btn_frame, text="Human vs Human", command=start_pvp,
             width=20, height=2, font=('Arial', 11)).pack(pady=10)
    
    tk.Button(btn_frame, text="Human vs AI", command=start_pvai,
             width=20, height=2, font=('Arial', 11)).pack(pady=10)
    
    tk.Button(btn_frame, text="AI vs AI", command=start_aivai,
             width=20, height=2, font=('Arial', 11), bg='lightblue').pack(pady=10)
    
    tk.Button(btn_frame, text="Tournament Mode", command=start_tournament,
             width=20, height=2, font=('Arial', 11, 'bold'), 
             bg='orange', fg='white').pack(pady=10)
    
    # Info
    info_text = """
    Features:
    • Multiple AI Types (Random, Greedy, Minimax)
    • Alpha-Beta Pruning Optimization
    • Live Statistics & Performance Metrics
    • Tournament System with Win Rates
    • Adjustable AI Speed
    """
    
    info_label = tk.Label(root, text=info_text, font=('Arial', 9), justify=tk.LEFT)
    info_label.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()