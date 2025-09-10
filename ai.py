from board import Board
from copy import deepcopy
from random import choice

INF = 10**9

class AI:
    def __init__(self, color, depth=6):
        # 'color' is the color this AI will play with (B or W)
        self.color = color
        self.depth = depth

    def alphabeta(self, current_board, is_maximizing, depth, turn, alpha, beta):
        # Abbruch: Tiefe 0 oder Endstellung
        if depth == 0 or current_board.get_winner() is not None:
            return self.get_value(current_board)

        next_turn = 'B' if turn == 'W' else 'W'
        board_color_up = current_board.get_color_up()
        pieces = current_board.get_pieces()

        # Züge für 'turn' erzeugen (mit Sprungregel & einfachem Ordering)
        flat_moves = []
        any_jump = False
        for idx, piece in enumerate(pieces):
            if piece.get_color() != turn:
                continue
            moves = piece.get_moves(current_board)
            if not moves:
                continue
            for mv in moves:
                flat_moves.append((idx, mv))
                if mv.get("eats_piece"):
                    any_jump = True

        # Wenn Sprünge existieren, nur Sprünge erlauben (Dame-Regel)
        if any_jump:
            flat_moves = [(i, m) for (i, m) in flat_moves if m.get("eats_piece")]

        # Move-Ordering: Sprünge zuerst (weitere Heuristiken möglich)
        flat_moves.sort(key=lambda im: 0 if im[1].get("eats_piece") else 1)

        # Keine legalen Züge -> wertet als terminal (Verlust für Spieler am Zug)
        if not flat_moves:
            return self.get_value(current_board)

        if is_maximizing:
            value = -INF
            for idx, mv in flat_moves:
                aux_board = Board(deepcopy(pieces), board_color_up)
                aux_board.move_piece(idx, int(mv["position"]))
                value = max(value, self.alphabeta(aux_board, False, depth - 1, next_turn, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:   # beta-cut
                    break
            return value
        else:
            value = INF
            for idx, mv in flat_moves:
                aux_board = Board(deepcopy(pieces), board_color_up)
                aux_board.move_piece(idx, int(mv["position"]))
                value = min(value, self.alphabeta(aux_board, True, depth - 1, next_turn, alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:   # alpha-cut
                    break
            return value

    def minimax(self, current_board, is_maximizing, depth, turn):
        if depth == 0 or current_board.get_winner() is not None:
            return self.get_value(current_board)

        next_turn = 'B' if turn == 'W' else 'W'
        board_color_up = current_board.get_color_up()
        current_pieces = current_board.get_pieces()
        piece_moves = list(map(lambda piece: piece.get_moves(current_board) if piece.get_color() == turn else False, current_pieces))

        if is_maximizing:
            maximum = -999
            for index, moves in enumerate(piece_moves):
                if moves == False:
                    continue
                for move in moves:
                    aux_board = Board(deepcopy(current_pieces), board_color_up)
                    aux_board.move_piece(index, int(move["position"]))
                    maximum = max(self.minimax(aux_board, False, depth - 1, next_turn), maximum)
            return maximum
        else:
            minimum = 999
            for index, moves in enumerate(piece_moves):
                if moves == False:
                    continue
                for move in moves:
                    aux_board = Board(deepcopy(current_pieces), board_color_up)
                    aux_board.move_piece(index, int(move["position"]))
                    minimum = min(self.minimax(aux_board, True, depth - 1, next_turn), minimum)
            return minimum

    def get_move(self, current_board):
        # Receives a Board object, returns the move it finds best suited.
        board_color_up = current_board.get_color_up()
        current_pieces = current_board.get_pieces()
        next_turn = "W" if self.color == "B" else "B"
        player_pieces = list(map(lambda piece: piece if piece.get_color() == self.color else False, current_pieces))
        possible_moves = []
        move_scores = []

        for index, piece in enumerate(player_pieces):
            if piece == False:
                continue
            for move in piece.get_moves(current_board):
                possible_moves.append({"piece": index, "move": move})

        # If any jump move is available, only jump moves can be made (checkers rule).
        jump_moves = list(filter(lambda move: move["move"].get("eats_piece") == True, possible_moves))
        if len(jump_moves) != 0:
            possible_moves = jump_moves

        # Scoring: Alpha-Beta statt Minimax, Tiefe = self.depth
        for move in possible_moves:
            aux_board = Board(deepcopy(current_pieces), board_color_up)
            aux_board.move_piece(move["piece"], int(move["move"]["position"]))
            # KORRIGIERT: Nach dem AI-Zug ist der Gegner dran -> is_maximizing=False
            move_scores.append(self.alphabeta(aux_board, False, self.depth - 1, next_turn, -INF, INF))

        best_score = max(move_scores)
        best_moves = []
        for index, move in enumerate(possible_moves):
            if move_scores[index] == best_score:
                best_moves.append(move)

        move_chosen = choice(best_moves)
        return {
            "position_to": move_chosen["move"]["position"],
            "position_from": player_pieces[move_chosen["piece"]].get_position()
        }

    def get_value(self, board):
        """Bewertungsfunktion für Dame/Checkers."""
        pieces = board.get_pieces()

        # Gewinn prüfen
        winner = board.get_winner()
        if winner is not None:
            return 1000 if winner == self.color else -1000

        player_score = 0.0
        opponent_score = 0.0

        for piece in pieces:
            # Grundwert: Dame (King) höher gewichtet als Man
            value = 5.0 if piece.is_king() else 3.0

            # KORRIGIERT: Position korrekt extrahieren
            pos = piece.get_position()
            if isinstance(pos, tuple):
                row, col = pos
            elif isinstance(pos, (int, float)):
                # Falls Position als einzelner Wert zurückgegeben wird
                # Annahme: 8x8 Brett, Position 0-63
                row = int(pos) // 8
                col = int(pos) % 8
            else:
                # Falls Position als String zurückgegeben wird (z.B. "A1", "B2")
                # Konvertiere zu int oder überspringe die Bewertung
                try:
                    pos_int = int(pos)
                    row = pos_int // 8
                    col = pos_int % 8
                except (ValueError, TypeError):
                    # Fallback: neutrale Position verwenden
                    row, col = 4, 4

            color = piece.get_color()

            # Positionsbonus: Nähe zur Kronungsreihe
            # Annahme: Reihe 0 oben, Reihe 7 unten
            if color == "B":          # B krönt oben (0)
                value += (7 - row) * 0.2
            else:                     # W krönt unten (7)
                value += row * 0.2

            # Zentrums­kontrolle
            if 2 <= row <= 5 and 2 <= col <= 5:
                value += 0.5

            # Back-Row-Verteidigung
            if (color == "B" and row == 7) or (color == "W" and row == 0):
                value += 0.3

            # Zuordnung zum jeweiligen Spieler
            if color == self.color:
                player_score += value
            else:
                opponent_score += value

        # Mobilität (Anzahl legaler Züge)
        player_moves = sum(len(p.get_moves(board) or []) for p in pieces if p.get_color() == self.color)
        opponent_moves = sum(len(p.get_moves(board) or []) for p in pieces if p.get_color() != self.color)

        player_score += 0.1 * player_moves
        opponent_score += 0.1 * opponent_moves

        return player_score - opponent_score
