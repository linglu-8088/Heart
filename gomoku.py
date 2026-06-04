import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import random
import math
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import copy
import numpy as np
import os

MODEL_DIR = "gomoku_models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class ValueNetwork(nn.Module):
    def __init__(self, input_size=225, hidden_size=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = x.view(-1, 225)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        return x


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, value):
        self.buffer.append((state.detach().cpu().clone(), float(value)))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, values = zip(*batch)
        states = torch.stack(states).to(device)
        values = torch.tensor(values, dtype=torch.float32, device=device).view(-1, 1)
        return states, values
    
    def __len__(self):
        return len(self.buffer)


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


class NeuralNetworkAI:
    def __init__(self):
        self.model = ValueNetwork().to(device)
        self.target_model = ValueNetwork().to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(capacity=50000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
    
    def board_to_tensor(self, board, player):
        tensor = torch.zeros(225, dtype=torch.float32)
        for i in range(15):
            for j in range(15):
                idx = i * 15 + j
                if board[i][j] == player:
                    tensor[idx] = 1.0
                elif board[i][j] != 0:
                    tensor[idx] = -1.0
        return tensor.to(device)
    
    def predict(self, board, player):
        with torch.no_grad():
            state = self.board_to_tensor(board, player)
            value = self.model(state)
            return value.item()
    
    def get_move_value(self, board, x, y, player):
        temp_board = [row[:] for row in board]
        temp_board[y][x] = player
        return self.predict(temp_board, player)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return 0
        
        states, targets = self.replay_buffer.sample(batch_size)
        states = states.to(device)
        targets = targets.to(device)
        
        predictions = self.model(states)
        loss = nn.MSELoss()(predictions, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def copy_from(self, other_ai):
        self.model.load_state_dict(other_ai.model.state_dict())
        self.target_model.load_state_dict(other_ai.target_model.state_dict())
    
    def save(self, path):
        if torch.cuda.is_available():
            self.model.cpu()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        if torch.cuda.is_available():
            self.model.cuda()
    
    def load(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.target_model.load_state_dict(self.model.state_dict())
        if torch.cuda.is_available():
            self.model.cuda()
            self.target_model.cuda()


class GobangGame:
    def __init__(self, root):
        self.root = root
        self.root.title("五子棋 - PyTorch神经网络版")
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
        
        self.nn_ai1 = NeuralNetworkAI()
        self.nn_ai2 = NeuralNetworkAI()
        self.train_games = 0
        self.game_count = 0
        self.training = False
        self.train_thread = None
        
        self.best_model_path = None
        self.best_model_wins = 0
        
        best_model_file = os.path.join(MODEL_DIR, "best_model.pt")
        if os.path.exists(best_model_file):
            self.nn_ai1.load(best_model_file)
            self.nn_ai2.load(best_model_file)
            print(f"已加载最佳模型: {best_model_file}")
        
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
        mode_window.geometry("380x300")
        mode_window.resizable(False, False)
        mode_window.transient(self.root)
        mode_window.grab_set()
        
        tk.Label(mode_window, text="请选择游戏模式", font=('Arial', 16), pady=15).pack()
        
        def select_pvp():
            self.game_mode = 0
            mode_window.destroy()
            self.update_status()
        
        def select_pve():
            self.game_mode = 1
            mode_window.destroy()
            self.show_difficulty_select()
        
        def select_ai_ai():
            self.game_mode = 2
            mode_window.destroy()
            self.show_ai_ai_options()
        
        tk.Button(mode_window, text="双人对战", font=('Arial', 13), bg='#90EE90', 
                  width=20, height=2, command=select_pvp).pack(pady=5)
        tk.Button(mode_window, text="人机对战", font=('Arial', 13), bg='#FFB6C1', 
                  width=20, height=2, command=select_pve).pack(pady=5)
        tk.Button(mode_window, text="双机对战 (训练神经网络)", font=('Arial', 13), bg='#87CEEB', 
                  width=20, height=2, command=select_ai_ai).pack(pady=5)
    
    def show_ai_ai_options(self):
        opt_window = tk.Toplevel(self.root)
        opt_window.title("双机对战选项")
        opt_window.geometry("400x380")
        opt_window.resizable(False, False)
        opt_window.transient(self.root)
        opt_window.grab_set()
        
        tk.Label(opt_window, text="PyTorch 神经网络训练", font=('Arial', 16), pady=10).pack()
        
        tk.Label(opt_window, text="训练局数:", font=('Arial', 12)).pack()
        games_var = tk.StringVar(value="50")
        games_entry = tk.Entry(opt_window, textvariable=games_var, font=('Arial', 12), width=10)
        games_entry.pack(pady=5)
        
        tk.Label(opt_window, text="(0 = 无限训练)", font=('Arial', 10), fg='gray').pack()
        
        tk.Label(opt_window, text="自动保存间隔(局):", font=('Arial', 12)).pack()
        save_var = tk.StringVar(value="10")
        save_entry = tk.Entry(opt_window, textvariable=save_var, font=('Arial', 12), width=10)
        save_entry.pack(pady=5)
        
        info_text = tk.Label(opt_window, 
            text="训练过程中:\n• 两个AI互相博弈\n• 胜者经验存入记忆库\n• 每局后训练网络\n• 网络会逐渐变强\n• 模型保存到 gomoku_models/",
            font=('Arial', 9), fg='#666', justify=tk.LEFT)
        info_text.pack(pady=10)
        
        btn_frame = tk.Frame(opt_window)
        btn_frame.pack(pady=10)
        
        self.auto_save_interval = 10
        
        def start_training():
            try:
                self.train_games = int(games_var.get()) if games_var.get() != "0" else 0
                self.auto_save_interval = int(save_var.get()) if save_var.get() != "0" else 10
            except:
                self.train_games = 50
                self.auto_save_interval = 10
            opt_window.destroy()
            self.start_ai_vs_ai()
        
        def start_once():
            self.train_games = 1
            self.auto_save_interval = 999999
            opt_window.destroy()
            self.start_ai_vs_ai()
        
        def load_model():
            filename = filedialog.askopenfilename(
                initialdir=MODEL_DIR,
                title="选择模型文件",
                filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
            )
            if filename:
                self.nn_ai1.load(filename)
                self.nn_ai2.load(filename)
                messagebox.showinfo("加载成功", f"已加载模型: {filename}")
        
        tk.Button(btn_frame, text="加载模型", font=('Arial', 10), bg='#87CEEB', 
                  width=10, command=load_model).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="开始训练", font=('Arial', 11), bg='#98FB98', 
                  width=10, command=start_training).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="下一局", font=('Arial', 10), bg='#FFD700', 
                  width=8, command=start_once).pack(side=tk.LEFT, padx=5)
    
    def start_ai_vs_ai(self):
        self.game_mode = 2
        self.training = True
        self.game_count = 0
        self.update_status()
        self.train_thread = threading.Thread(target=self.run_training_loop)
        self.train_thread.start()
    
    def run_training(self):
        ai1_wins = 0
        ai2_wins = 0
        draws = 0
        
        while self.training and (self.train_games == 0 or self.game_count < self.train_games):
            winner = self.play_game()
            self.game_count += 1
            
            if winner == 1:
                ai1_wins += 1
            elif winner == 2:
                ai2_wins += 1
            else:
                draws += 1
            
            state_ai1 = self.nn_ai1.board_to_tensor(self.board, 1)
            state_ai2 = self.nn_ai2.board_to_tensor(self.board, 2)
            
            if winner == 1:
                self.nn_ai1.replay_buffer.push(state_ai1, 1.0)
                self.nn_ai2.replay_buffer.push(state_ai2, -1.0)
            elif winner == 2:
                self.nn_ai1.replay_buffer.push(state_ai1, -1.0)
                self.nn_ai2.replay_buffer.push(state_ai2, 1.0)
            
            loss = self.nn_ai1.train_step(32)
            
            self.root.after(0, self.update_training_ui, winner)
            
            if self.game_count % self.auto_save_interval == 0:
                win_rate = ai1_wins / self.game_count if self.game_count > 0 else 0
                
                if win_rate > self.best_model_wins:
                    best_path = os.path.join(MODEL_DIR, "best_model.pt")
                    self.nn_ai1.save(best_path)
                    self.best_model_wins = win_rate
                    self.root.after(0, lambda w=win_rate: self.status_label.config(
                        text=f"新最佳模型! 胜率: {w*100:.1f}%"))
            
            self.board = [[0] * self.board_size for _ in range(self.board_size)]
            self.current_player = 1
            self.last_move = None
            
            self.nn_ai1.decay_epsilon()
            self.nn_ai2.decay_epsilon()
        
        self.root.after(0, self.training_finished)
    
    def run_training_loop(self):
        ai1_wins = 0
        ai2_wins = 0
        draws = 0
        
        try:
            while self.training and (self.train_games == 0 or self.game_count < self.train_games):
                winner, final_board = self.play_game_training()
                self.game_count += 1
                
                if winner == 1:
                    ai1_wins += 1
                elif winner == 2:
                    ai2_wins += 1
                else:
                    draws += 1
                
                state_ai1 = self.nn_ai1.board_to_tensor(final_board, 1)
                state_ai2 = self.nn_ai2.board_to_tensor(final_board, 2)
                
                if winner == 1:
                    self.nn_ai1.replay_buffer.push(state_ai1, 1.0)
                    self.nn_ai2.replay_buffer.push(state_ai2, -1.0)
                elif winner == 2:
                    self.nn_ai1.replay_buffer.push(state_ai1, -1.0)
                    self.nn_ai2.replay_buffer.push(state_ai2, 1.0)
                else:
                    self.nn_ai1.replay_buffer.push(state_ai1, 0.0)
                    self.nn_ai2.replay_buffer.push(state_ai2, 0.0)
                
                self.nn_ai1.train_step(32)
                self.nn_ai2.train_step(32)
                
                self.board = [row[:] for row in final_board]
                self.root.after(0, self.update_training_ui, winner)
                
                if self.game_count % self.auto_save_interval == 0:
                    win_rate = ai1_wins / self.game_count if self.game_count > 0 else 0
                    
                    if win_rate > self.best_model_wins:
                        best_path = os.path.join(MODEL_DIR, "best_model.pt")
                        self.nn_ai1.save(best_path)
                        self.best_model_wins = win_rate
                        self.root.after(0, lambda w=win_rate: self.status_label.config(
                            text=f"鏂版渶浣虫ā鍨? 鑳滅巼: {w*100:.1f}%"))
                
                self.board = [[0] * self.board_size for _ in range(self.board_size)]
                self.current_player = 1
                self.last_move = None
                
                self.nn_ai1.decay_epsilon()
                self.nn_ai2.decay_epsilon()
            
            self.root.after(0, self.training_finished)
        except Exception:
            error_text = traceback.format_exc()
            self.root.after(0, lambda msg=error_text: self.training_failed(msg))
    
    def play_game_training(self):
        board = [[0] * self.board_size for _ in range(self.board_size)]
        current = 1
        
        for _ in range(225):
            if current == 1:
                move = self.get_nn_move(board, 1, self.nn_ai1)
            else:
                move = self.get_nn_move(board, 2, self.nn_ai2)
            
            if move is None:
                return 0, board
            
            x, y = move
            board[y][x] = current
            
            if self.check_win(board, x, y):
                return current, board
            
            current = 3 - current
        
        return 0, board
    
    def play_game(self):
        board = [[0] * self.board_size for _ in range(self.board_size)]
        current = 1
        
        for _ in range(225):
            if current == 1:
                move = self.get_nn_move(board, 1, self.nn_ai1)
            else:
                move = self.get_nn_move(board, 2, self.nn_ai2)
            
            if move is None:
                return 0
            
            x, y = move
            board[y][x] = current
            
            if self.check_win(board, x, y):
                return current
            
            current = 3 - current
        
        return 0
    
    def get_nn_move(self, board, player, nn_ai):
        candidates = self.get_candidate_moves_fast(board)
        
        if not candidates:
            return None
        
        for x, y in candidates:
            temp_board = [row[:] for row in board]
            temp_board[y][x] = player
            if self.check_win(temp_board, x, y):
                return (x, y)
        
        best_score = float('-inf')
        best_move = None
        
        opponent = 3 - player
        
        scored = []
        for x, y in candidates:
            score = nn_ai.get_move_value(board, x, y, player)
            scored.append((x, y, score))
        
        scored.sort(key=lambda t: t[2], reverse=True)
        
        for x, y, move_score in scored[:15]:
            opp_score = -nn_ai.get_move_value(board, x, y, opponent)
            score = move_score * 0.7 + opp_score * 0.3
            
            if score > best_score:
                best_score = score
                best_move = (x, y)
        
        return best_move
    
    def get_candidate_moves_fast(self, board):
        candidates = set()
        checked = set()
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] != 0:
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.board_size and 0 <= nj < self.board_size:
                                if board[ni][nj] == 0 and (ni, nj) not in checked:
                                    candidates.add((nj, ni))
                                    checked.add((ni, nj))
        
        if len(candidates) < 5:
            candidates = set()
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if board[i][j] == 0:
                        candidates.add((j, i))
        
        return list(candidates)
    
    def check_win(self, board, x, y):
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]
        
        player = board[y][x]
        
        for dir1, dir2 in directions:
            count = 1
            
            dx, dy = dir1
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            dx, dy = dir2
            nx, ny = x + dx, y + dy
            while 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[ny][nx] == player:
                count += 1
                nx += dx
                ny += dy
            
            if count >= 5:
                return True
        
        return False
    
    def update_training_ui(self, winner):
        winner_text = "AI1 (黑)" if winner == 1 else ("AI2 (白)" if winner == 2 else "平局")
        self.status_label.config(text=f"训练中... 第{self.game_count}局: {winner_text}")
    
    def training_finished(self):
        self.training = False
        if self.train_games > 0:
            messagebox.showinfo("训练完成", f"完成 {self.game_count} 局训练!\n经验池大小: {len(self.nn_ai1.replay_buffer)}")
    
    def training_failed(self, error_text):
        self.training = False
        self.status_label.config(text="训练已中断，请查看错误信息")
        messagebox.showerror("训练错误", error_text)
    
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
        if self.game_over or not self.show_hint or self.ai_thinking or self.game_mode == 2:
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
        if self.game_over or self.ai_thinking or self.game_mode == 2:
            return
        
        x = round((event.x - self.margin) / self.cell_size)
        y = round((event.y - self.margin) / self.cell_size)
        
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            if self.board[y][x] == 0:
                self.board[y][x] = self.current_player
                self.last_move = (x, y)
                self.hint_position = None
                self.draw_board()
                
                if self.check_win(self.board, x, y):
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
        
        if move and self.check_win(self.board, x, y):
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
        candidates = self.get_candidate_moves_fast(self.board)
        for x, y in candidates:
            self.board[y][x] = player
            if self.check_win(self.board, x, y):
                self.board[y][x] = 0
                return (x, y)
            self.board[y][x] = 0
        return None
    
    def find_double_threat(self, player):
        candidates = self.get_candidate_moves_fast(self.board)
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
        candidates = self.get_candidate_moves_fast(self.board)
        
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
        candidates = self.get_candidate_moves_fast(self.board)
        
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
        
        candidates = self.get_candidate_moves_fast(self.board)
        
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
    
    def toggle_hint(self):
        if self.game_mode == 1 and self.current_player == 2 or self.game_mode == 2:
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
        elif self.game_mode == 1:
            player_text = "你 (黑棋)" if self.current_player == 1 else "AI (白棋)"
            diff_text = ["", "简单", "中等", "困难"][self.ai_difficulty]
            status = f"人机对战({diff_text}) - 当前回合: {player_text}"
        else:
            if self.game_count > 0:
                status = f"双机训练中... 第{self.game_count}局"
            else:
                status = "双机对战 - 准备开始"
        
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
        self.training = False
        self.game_count = 0
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
    
    def save_current_model():
        filename = filedialog.asksaveasfilename(
            initialdir=MODEL_DIR,
            title="保存模型",
            defaultextension=".pt",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            game.nn_ai1.save(filename)
            messagebox.showinfo("保存成功", f"模型已保存到: {filename}")
    
    def load_model_menu():
        filename = filedialog.askopenfilename(
            initialdir=MODEL_DIR,
            title="加载模型",
            filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")]
        )
        if filename:
            game.nn_ai1.load(filename)
            game.nn_ai2.load(filename)
            messagebox.showinfo("加载成功", f"已加载模型: {filename}")
    
    game_menu.add_command(label="保存模型", command=save_current_model)
    game_menu.add_command(label="加载模型", command=load_model_menu)
    game_menu.add_separator()
    game_menu.add_command(label="退出", command=root.quit)
    menubar.add_cascade(label="游戏", menu=game_menu)
    root.config(menu=menubar)
    
    root.mainloop()


if __name__ == '__main__':
    main()
