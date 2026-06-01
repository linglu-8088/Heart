import tkinter as tk
from tkinter import messagebox
import threading

class ChessPattern:
    WIN = 10000000
    LIVE_FOUR = 1000000
    FOUR_FOUR = 500000
    FOUR_THREE = 400000
    LIVE_THREE = 100000
    THREE_THREE = 80000
    FOUR_ONE = 50000
    THREE_ONE = 10000
    LIVE_TWO = 5000
    TWO_TWO = 3000
    TWO_ONE = 1000
    LIVE_ONE = 500

SCORES = {}

class GobangGame:
    def __init__(self, root):
        self.root = root
        self.root.title("五子棋")
        self.root.resizable(False, False)
        
        self.board_size = 15
        self.cell_size = 40
        self.margin = 30
        self.piece_size = 16
        
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        self.last_move = None
        self.hint_position = None
        self.show_hint = False
        
        self.game_mode = 0
        self.ai_difficulty = 2
        self.ai_thinking = False
        
        self.ai_player = 2
        self.human_player = 1
        
        canvas_width = self.cell_size * (self.board_size - 1) + 2 * self.margin
        canvas_height = self.cell_size * (self.board_size - 1) + 2 * self.margin
        
        self.canvas = tk.Canvas(
            root, 
            width=canvas_width, 
            height=canvas_height,
            bg='#E3CD98'
        )
        self.canvas.pack()
        
        self.canvas.bind('<Button-1>', self.click_handler)
        self.canvas.bind('<Motion>', self.mouse_move)
        
        top_frame = tk.Frame(root, bg='#F5F5F5')
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = tk.Label(top_frame, text="", font=('Arial', 14), bg='#F5F5F5')
        self.status_label.pack(side=tk.LEFT)
        
        btn_frame = tk.Frame(top_frame, bg='#F5F5F5')
        btn_frame.pack(side=tk.RIGHT)
        
        self.hint_btn = tk.Button(btn_frame, text="💡 提示", command=self.toggle_hint, 
                                   bg='#FFD700', font=('Arial', 11))
        self.hint_btn.pack(side=tk.LEFT, padx=5)
        
        self.draw_board()
        self.show_mode_select()
        
    def show_mode_select(self):
        mode_window = tk.Toplevel(self.root)
        mode_window.title("选择游戏模式")
        mode_window.geometry("300x200")
        mode_window.resizable(False, False)
        mode_window.transient(self.root)
        mode_window.grab_set()
        
        tk.Label(mode_window, text="请选择游戏模式", font=('Arial', 16), pady=20).pack()
        
        def select_pvp():
            self.game_mode = 0
            mode_window.destroy()
            self.update_status()
        
        def select_pve():
            self.game_mode = 1
            mode_window.destroy()
            self.show_difficulty_select()
        
        tk.Button(mode_window, text="双人对战", font=('Arial', 14), bg='#90EE90', 
                  width=15, height=2, command=select_pvp).pack(pady=10)
        tk.Button(mode_window, text="人机对战", font=('Arial', 14), bg='#FFB6C1', 
                  width=15, height=2, command=select_pve).pack(pady=10)
    
    def show_difficulty_select(self):
        diff_window = tk.Toplevel(self.root)
        diff_window.title("选择难度")
        diff_window.geometry("300x200")
        diff_window.resizable(False, False)
        diff_window.transient(self.root)
        diff_window.grab_set()
        
        tk.Label(diff_window, text="请选择AI难度", font=('Arial', 16), pady=20).pack()
        
        def select_diff(level):
            self.ai_difficulty = level
            diff_window.destroy()
            self.update_status()
        
        tk.Button(diff_window, text="简单", font=('Arial', 14), bg='#98FB98', 
                  width=15, height=2, command=lambda: select_diff(1)).pack(pady=5)
        tk.Button(diff_window, text="中等", font=('Arial', 14), bg='#FFD700', 
                  width=15, height=2, command=lambda: select_diff(2)).pack(pady=5)
        tk.Button(diff_window, text="困难", font=('Arial', 14), bg='#FF6347', 
                  width=15, height=2, command=lambda: select_diff(3)).pack(pady=5)
    
    def draw_board(self):
        self.canvas.delete('all')
        
        for i in range(self.board_size):
            start = self.margin
            end = self.margin + self.cell_size * (self.board_size - 1)
            
            self.canvas.create_line(
                start, self.margin + i * self.cell_size,
                end, self.margin + i * self.cell_size,
                fill='black', width=1
            )
            
            self.canvas.create_line(
                self.margin + i * self.cell_size, start,
                self.margin + i * self.cell_size, end,
                fill='black', width=1
            )
        
        star_points = [(3, 3), (11, 3), (7, 7), (3, 11), (11, 11)]
        for x, y in star_points:
            cx = self.margin + x * self.cell_size
            cy = self.margin + y * self.cell_size
            r = 4
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill='black', outline='black'
            )
        
        self.draw_pieces()
        
        if self.last_move:
            x, y = self.last_move
            cx = self.margin + x * self.cell_size
            cy = self.margin + y * self.cell_size
            self.canvas.create_oval(
                cx - 6, cy - 6, cx + 6, cy + 6,
                fill='red', outline='red'
            )
        
        if self.hint_position:
            x, y = self.hint_position
            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[y][x] == 0:
                cx = self.margin + x * self.cell_size
                cy = self.margin + y * self.cell_size
                self.canvas.create_oval(
                    cx - self.piece_size, cy - self.piece_size,
                    cx + self.piece_size, cy + self.piece_size,
                    fill='', outline='green', width=3
                )
        
        if self.ai_thinking:
            self.canvas.create_text(
                self.margin + self.cell_size * 7,
                self.margin + self.cell_size * 7,
                text="AI思考中...",
                font=('Arial', 20),
                fill='red',
                tags='thinking'
            )
    
    def draw_pieces(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    cx = self.margin + j * self.cell_size
                    cy = self.margin + i * self.cell_size
                    
                    if self.board[i][j] == 1:
                        color = 'black'
                    else:
                        color = 'white'
                    
                    self.canvas.create_oval(
                        cx - self.piece_size, cy - self.piece_size,
                        cx + self.piece_size, cy + self.piece_size,
                        fill=color, outline=color
                    )
    
    def mouse_move(self, event):
        if self.game_over or not self.show_hint or self.ai_thinking:
            return
        
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.board[y][x] == 0:
                if self.hint_position != (x, y):
                    self.hint_position = (x, y)
                    self.draw_board()
            else:
                self.hint_position = None
                self.draw_board()
        else:
            self.hint_position = None
            self.draw_board()
    
    def click_handler(self, event):
        if self.game_over or self.ai_thinking:
            return
        
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.board[y][x] == 0:
                self.board[y][x] = self.current_player
                self.last_move = (x, y)
                self.hint_position = None
                self.draw_board()
                
                if self.check_win(x, y):
                    winner = "你" if self.current_player == 1 else "白棋"
                    messagebox.showinfo("游戏结束", f"{winner}获胜！")
                    self.game_over = True
                    return
                
                if self.is_board_full():
                    messagebox.showinfo("游戏结束", "平局！")
                    self.game_over = True
                    return
                
                if self.game_mode == 1 and self.current_player == 1:
                    self.current_player = 2
                    self.update_status()
                    self.root.after(100, self.ai_move)
                else:
                    self.current_player = 3 - self.current_player
                    self.update_status()
    
    def is_board_full(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    return False
        return True
    
    def ai_move(self):
        self.ai_thinking = True
        self.draw_board()
        
        thread = threading.Thread(target=self._ai_move_thread)
        thread.start()
    
    def _ai_move_thread(self):
        move = self.get_ai_move()
        
        if move:
            x, y = move
            self.board[y][x] = 2
            self.last_move = (x, y)
        
        self.ai_thinking = False
        
        if move and self.check_win(x, y):
            self.root.after(0, self._show_win_message)
        elif self.is_board_full():
            self.root.after(0, self._show_draw_message)
        else:
            self.root.after(0, self._finish_ai_move)
    
    def _show_win_message(self):
        messagebox.showinfo("游戏结束", "AI (白棋) 获胜！")
        self.game_over = True
    
    def _show_draw_message(self):
        messagebox.showinfo("游戏结束", "平局！")
        self.game_over = True
    
    def _finish_ai_move(self):
        self.current_player = 1
        self.update_status()
        self.draw_board()
    
    def get_ai_move(self):
        piece_count = sum(1 for i in range(self.board_size) for j in range(self.board_size) if self.board[i][j] != 0)
        
        if piece_count == 0:
            return (7, 7)
        
        if piece_count == 1:
            player_pos = None
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 1:
                        player_pos = (j, i)
                        break
            if player_pos:
                moves = [(7, 7), (6, 6), (6, 7), (7, 6), (8, 7), (7, 8), (8, 8), (5, 7), (7, 5)]
                for mx, my in moves:
                    if 0 <= mx < self.board_size and 0 <= my < self.board_size:
                        if self.board[my][mx] == 0:
                            return (mx, my)
        
        if self.ai_difficulty == 1:
            return self.get_simple_move()
        
        win_move = self.find_winning_move(self.ai_player)
        if win_move:
            return win_move
        
        block_move = self.find_winning_move(self.human_player)
        if block_move:
            return block_move
        
        double_threat = self.find_double_threat(self.ai_player)
        if double_threat:
            return double_threat
        
        double_threat = self.find_double_threat(self.human_player)
        if double_threat:
            return double_threat
        
        depth = {1: 2, 2: 3, 3: 4}.get(self.ai_difficulty, 3)
        return self.get_best_move(depth)
    
    def find_winning_move(self, player):
        candidates = self.get_candidate_moves_fast()
        for x, y in candidates:
            self.board[y][x] = player
            if self.check_win(x, y):
                self.board[y][x] = 0
                return (x, y)
            self.board[y][x] = 0
        return None
    
    def find_double_threat(self, player):
        candidates = self.get_candidate_moves_fast()
        threats = []
        
        for x, y in candidates:
            self.board[y][x] = player
            count = self.count_threats(x, y, player)
            self.board[y][x] = 0
            
            if count >= 2:
                threats.append((x, y, count))
        
        if threats:
            threats.sort(key=lambda t: t[2], reverse=True)
            return (threats[0][0], threats[0][1])
        return None
    
    def count_threats(self, x, y, player):
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dx, dy in directions:
            line_count = self.count_line_threats(x, y, dx, dy, player)
            if line_count >= 4:
                count += 2
            elif line_count >= 3:
                count += 1
        
        return count
    
    def count_line_threats(self, x, y, dx, dy, player):
        count = 0
        
        nx, ny = x + dx, y + dy
        while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
            count += 1
            nx += dx
            ny += dy
        
        nx, ny = x - dx, y - dy
        while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
            count += 1
            nx -= dx
            ny -= dy
        
        return count
    
    def get_simple_move(self):
        candidates = self.get_candidate_moves_fast()
        
        best_score = float('-inf')
        best_move = None
        
        for x, y in candidates:
            attack = self.evaluate_position(x, y, self.ai_player)
            defense = self.evaluate_position(x, y, self.human_player)
            score = attack * 2 + defense * 3
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
        
        return best_move
    
    def get_best_move(self, depth):
        candidates = self.get_candidate_moves_fast()
        
        best_score = float('-inf')
        best_move = None
        alpha = float('-inf')
        beta = float('inf')
        
        scored_candidates = []
        for x, y in candidates:
            attack = self.evaluate_position(x, y, self.ai_player)
            defense = self.evaluate_position(x, y, self.human_player)
            score = attack * 2 + defense * 3
            scored_candidates.append((x, y, score))
        
        scored_candidates.sort(key=lambda t: t[2], reverse=True)
        top_candidates = scored_candidates[:10]
        
        for x, y, _ in top_candidates:
            self.board[y][x] = self.ai_player
            score = self.minimax(depth - 1, alpha, beta, False)
            self.board[y][x] = 0
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
            
            alpha = max(alpha, best_score)
        
        return best_move
    
    def minimax(self, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return self.evaluate_board()
        
        candidates = self.get_candidate_moves_fast()
        
        if not candidates:
            return self.evaluate_board()
        
        scored_candidates = []
        for x, y in candidates:
            attack = self.evaluate_position(x, y, self.ai_player)
            defense = self.evaluate_position(x, y, self.human_player)
            score = attack * 2 + defense * 3
            scored_candidates.append((x, y, score))
        
        scored_candidates.sort(key=lambda t: t[2], reverse=True)
        
        if is_maximizing:
            max_eval = float('-inf')
            for x, y, _ in scored_candidates[:8]:
                self.board[y][x] = self.ai_player
                eval_score = self.minimax(depth - 1, alpha, beta, False)
                self.board[y][x] = 0
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for x, y, _ in scored_candidates[:8]:
                self.board[y][x] = self.human_player
                eval_score = self.minimax(depth - 1, alpha, beta, True)
                self.board[y][x] = 0
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_candidate_moves_fast(self):
        candidates = set()
        checked = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] != 0:
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                                if self.board[ni][nj] == 0 and (ni, nj) not in checked:
                                    candidates.add((nj, ni))
                                    checked.add((ni, nj))
        
        if len(candidates) < 5:
            candidates = set()
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] == 0:
                        candidates.add((j, i))
        
        return list(candidates)
    
    def evaluate_board(self):
        score = 0
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == self.ai_player:
                    score += self.evaluate_position(j, i, self.ai_player)
                elif self.board[i][j] == self.human_player:
                    score -= self.evaluate_position(j, i, self.human_player)
        
        return score
    
    def evaluate_position(self, x, y, player):
        if self.board[y][x] != 0:
            return -100000
        
        opponent = 3 - player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        total_score = 0
        four_count = 0
        three_count = 0
        
        for dx, dy in directions:
            line_info = self.analyze_line(x, y, dx, dy, player)
            total_score += line_info['score']
            
            if line_info['four']:
                four_count += 1
            if line_info['three']:
                three_count += 1
        
        if four_count >= 2:
            total_score += ChessPattern.FOUR_FOUR
        elif four_count >= 1 and three_count >= 1:
            total_score += ChessPattern.FOUR_THREE
        
        opponent_four = 0
        opponent_three = 0
        for dx, dy in directions:
            line_info = self.analyze_line(x, y, dx, dy, opponent)
            if line_info['four']:
                opponent_four += 1
            if line_info['three']:
                opponent_three += 1
        
        if opponent_four >= 2:
            total_score -= ChessPattern.FOUR_FOUR
        elif opponent_four >= 1 and opponent_three >= 1:
            total_score -= ChessPattern.FOUR_THREE
        
        return total_score
    
    def analyze_line(self, x, y, dx, dy, player):
        line = []
        opponent = 3 - player
        
        for i in range(4, -1, -1):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                line.append(self.board[ny][nx])
            else:
                line.append(opponent)
        
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                line.append(self.board[ny][nx])
            else:
                line.append(opponent)
        
        max_score = 0
        has_four = False
        has_three = False
        
        for i in range(len(line) - 4):
            window = line[i:i+5]
            info = self.evaluate_window(window, player)
            max_score = max(max_score, info['score'])
            if info['four']:
                has_four = True
            if info['three']:
                has_three = True
        
        return {'score': max_score, 'four': has_four, 'three': has_three}
    
    def evaluate_window(self, window, player):
        opponent = 3 - player
        player_count = window.count(player)
        empty_count = window.count(0)
        opponent_count = window.count(opponent)
        
        result = {'score': 0, 'four': False, 'three': False}
        
        if opponent_count > 0:
            return result
        
        if player_count == 5:
            result['score'] = ChessPattern.WIN
            result['four'] = True
        elif player_count == 4:
            if empty_count == 1:
                result['score'] = ChessPattern.LIVE_FOUR
                result['four'] = True
            else:
                result['score'] = ChessPattern.FOUR_ONE
                result['four'] = True
        elif player_count == 3:
            if empty_count == 2:
                result['score'] = ChessPattern.LIVE_THREE
                result['three'] = True
            elif empty_count == 1:
                result['score'] = ChessPattern.THREE_ONE
                result['three'] = True
        elif player_count == 2:
            if empty_count == 3:
                result['score'] = ChessPattern.LIVE_TWO
            elif empty_count == 2:
                result['score'] = ChessPattern.TWO_ONE
        elif player_count == 1:
            if empty_count == 4:
                result['score'] = ChessPattern.LIVE_ONE
        
        return result
    
    def check_win(self, x, y):
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        
        player = self.board[y][x]
        
        for dir1, dir2 in directions:
            count = 1
            
            dx, dy = dir1
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            dx, dy = dir2
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            if count >= 5:
                return True
        
        return False
    
    def toggle_hint(self):
        if self.game_mode == 1 and self.current_player == 2:
            return
        
        self.show_hint = not self.show_hint
        if not self.show_hint:
            self.hint_position = None
            self.draw_board()
        self.hint_btn.config(text="💡 提示" if not self.show_hint else "🔍 提示中")
    
    def update_status(self):
        if self.game_mode == 0:
            player_text = "黑棋" if self.current_player == 1 else "白棋"
            status = f"双人对战 - 当前回合: {player_text}"
        else:
            player_text = "你 (黑棋)" if self.current_player == 1 else "AI (白棋)"
            diff_text = ["", "简单", "中等", "困难"][self.ai_difficulty]
            status = f"人机对战({diff_text}) - 当前回合: {player_text}"
        
        self.status_label.config(text=status)
        self.canvas.delete('last_indicator')
    
    def restart(self):
        self.board = [[0] * self.board_size for _ in range(self.board_size)]
        self.current_player = 1
        self.game_over = False
        self.last_move = None
        self.hint_position = None
        self.show_hint = False
        self.ai_thinking = False
        self.hint_btn.config(text="💡 提示")
        self.draw_board()
        self.update_status()

def main():
    root = tk.Tk()
    game = GobangGame(root)
    
    menubar = tk.Menu(root)
    game_menu = tk.Menu(menubar, tearoff=0)
    game_menu.add_command(label="重新开始", command=game.restart)
    game_menu.add_separator()
    game_menu.add_command(label="选择模式", command=game.show_mode_select)
    game_menu.add_separator()
    game_menu.add_command(label="退出", command=root.quit)
    menubar.add_cascade(label="游戏", menu=game_menu)
    root.config(menu=menubar)
    
    root.mainloop()

if __name__ == '__main__':
    main()