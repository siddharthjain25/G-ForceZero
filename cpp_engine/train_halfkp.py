import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import numpy as np
import random
import sys
import os

NUM_FEATURES = 40960 # 64 (king squares) * 10 (piece types) * 64 (piece squares)

def get_piece_idx(piece_type, is_mine):
    pt = piece_type - 1 # Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4
    return pt if is_mine else pt + 5

def fen_to_halfkp(fen):
    board = chess.Board(fen)
    w_feat = []
    b_feat = []
    
    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)
    if wk_sq is None or bk_sq is None:
        return [], []
        
    bk_sq_flipped = bk_sq ^ 56
    
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
            
        sq_w = sq
        sq_b = sq ^ 56
        
        # White perspective
        is_mine_w = (piece.color == chess.WHITE)
        pt_idx_w = get_piece_idx(piece.piece_type, is_mine_w)
        idx_w = pt_idx_w * 4096 + wk_sq * 64 + sq_w
        w_feat.append(idx_w)
        
        # Black perspective
        is_mine_b = (piece.color == chess.BLACK)
        pt_idx_b = get_piece_idx(piece.piece_type, is_mine_b)
        idx_b = pt_idx_b * 4096 + bk_sq_flipped * 64 + sq_b
        b_feat.append(idx_b)
        
    return w_feat, b_feat

class HalfKPNet(nn.Module):
    def __init__(self):
        super(HalfKPNet, self).__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, 256)
        self.fc2 = nn.Linear(512, 1)
        
    def forward(self, w_features, b_features):
        w_acc = torch.clamp(self.fc1(w_features), 0.0, 1.0)
        b_acc = torch.clamp(self.fc1(b_features), 0.0, 1.0)
        acc = torch.cat([w_acc, b_acc], dim=1)
        out = self.fc2(acc)
        return torch.sigmoid(out)

class ChessDataset(Dataset):
    def __init__(self, epd_file):
        self.w_features = []
        self.b_features = []
        self.scores = []
        
        print(f"Loading HalfKP dataset from {epd_file}...")
        with open(epd_file, 'r') as f:
            lines = f.readlines()
            
        random.shuffle(lines)
        for i, line in enumerate(lines):
            parts = line.split('"')
            if len(parts) >= 2:
                fen = parts[0].replace(" c9 ", "").strip()
                cp = float(parts[1])
                
                w_feat, b_feat = fen_to_halfkp(fen)
                if not w_feat: continue
                
                prob = 1.0 / (1.0 + np.exp(-0.003 * cp))
                self.w_features.append(w_feat)
                self.b_features.append(b_feat)
                self.scores.append(prob)
                
            if (i+1) % 100000 == 0:
                print(f"Loaded {i+1} positions...")
                
        print(f"Total valid positions: {len(self.scores)}")

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        w_tensor = torch.zeros(NUM_FEATURES, dtype=torch.float32)
        if self.w_features[idx]:
            w_tensor[self.w_features[idx]] = 1.0
            
        b_tensor = torch.zeros(NUM_FEATURES, dtype=torch.float32)
        if self.b_features[idx]:
            b_tensor[self.b_features[idx]] = 1.0
            
        return w_tensor, b_tensor, torch.tensor([self.scores[idx]], dtype=torch.float32)

def train(dataset_file, epochs=2):
    dataset = ChessDataset(dataset_file)
    
    # Train / Val Split (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False, num_workers=0)
    
    model = HalfKPNet()
    
    # Initialize biases to 0 for better stability
    nn.init.zeros_(model.fc1.bias)
    nn.init.zeros_(model.fc2.bias)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print("Starting NNUE training...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss = 0
        for i, (w_batch, b_batch, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(w_batch, b_batch)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
                
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for w_batch, b_batch, labels in val_loader:
                outputs = model(w_batch, b_batch)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"--- Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ---")
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model...")
            torch.save(model.state_dict(), "best_model.pth")
            export_quantized_weights(model, "nnue_weights.bin")
            
    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

def export_quantized_weights(model, filename):
    fc1_w = (model.fc1.weight.detach().cpu().numpy() * 256).astype('int16')
    fc1_b = (model.fc1.bias.detach().cpu().numpy() * 256).astype('int16')
    
    fc2_w = (model.fc2.weight.detach().cpu().numpy() * 64).astype('int16')
    fc2_b = (model.fc2.bias.detach().cpu().numpy() * 64 * 256).astype('int32')
    
    with open(filename, "wb") as f:
        f.write(fc1_w.tobytes())
        f.write(fc1_b.tobytes())
        f.write(fc2_w.tobytes())
        f.write(fc2_b.tobytes())
        
    print(f"Quantized NNUE weights exported to {filename}")

if __name__ == "__main__":
    epd_file = sys.argv[1] if len(sys.argv) > 1 else "stockfish_dataset.epd"
    if not os.path.exists(epd_file):
        print(f"Dataset {epd_file} not found!")
        sys.exit(1)
    train(epd_file)
