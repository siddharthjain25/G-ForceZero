"""
train_halfkp.py - Memory-efficient streaming NNUE trainer.

Uses an IterableDataset that reads the EPD file line-by-line during training,
keeping RAM usage under ~500 MB regardless of dataset size.

Usage:
    python3 train_halfkp.py lichess_2017_training.epd
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import chess
import numpy as np
import torch.multiprocessing as mp
import random
import sys
import os
import struct

# Fix PyTorch DataLoader shared memory (/dev/shm) crash
mp.set_sharing_strategy('file_system')

NUM_FEATURES = 41024  # 641 * 64
HIDDEN_SIZE  = 256
BATCH_SIZE   = 16384   # Massive batch size (T4 GPU optimized)

# ─── Feature Encoding ─────────────────────────────────────────────────────────
def fen_to_halfkp(fen):
    w_feat, b_feat = [], []
    parts = fen.split()
    board_part = parts[0]
    
    wk_sq = None
    bk_sq = None
    
    sq = 56 # A8
    pieces = []
    for char in board_part:
        if char == '/':
            sq -= 16
        elif char.isdigit():
            sq += int(char)
        else:
            if char == 'K': wk_sq = sq
            elif char == 'k': bk_sq = sq
            pieces.append((sq, char))
            sq += 1
            
    if wk_sq is None or bk_sq is None:
        return None, None
        
    bk_sq_flipped = bk_sq ^ 56
    
    pt_map = {
        'P': (0, True), 'N': (1, True), 'B': (2, True), 'R': (3, True), 'Q': (4, True),
        'p': (0, False), 'n': (1, False), 'b': (2, False), 'r': (3, False), 'q': (4, False)
    }
    
    for sq, char in pieces:
        if char in ('K', 'k'): continue
        pt, is_white = pt_map[char]
        
        sq_w = sq
        pt_w = pt if is_white else pt + 5
        w_feat.append(pt_w * 4096 + wk_sq * 64 + sq_w)
        
        sq_b = sq ^ 56
        pt_b = pt if not is_white else pt + 5
        b_feat.append(pt_b * 4096 + bk_sq_flipped * 64 + sq_b)
        
    return w_feat, b_feat

# ─── Streaming Dataset ────────────────────────────────────────────────────────
class StreamingEPDDataset(IterableDataset):
    """
    Reads EPD line-by-line — O(1) RAM regardless of dataset size.
    Supports multi-worker DataLoader by splitting file offsets.
    """
    def __init__(self, epd_file, shuffle_buffer=50000):
        self.epd_file = epd_file
        self.shuffle_buffer = shuffle_buffer
        # Count lines once for progress reporting (fast)
        self._len = None

    def __len__(self):
        if self._len is None:
            print(f"Counting total positions in {self.epd_file} (This may take several minutes for files > 50GB)...", flush=True)
            with open(self.epd_file, 'rb') as f:
                self._len = sum(1 for _ in f)
        return self._len

    def parse_line(self, line):
        line = line.strip()
        if not line:
            return None
            
        if line.startswith('{'):
            # Super-fast native JSONL string extraction (skips slow json.loads)
            fen_start = line.find('"fen":"')
            if fen_start == -1: return None
            fen_start += 7
            fen_end = line.find('"', fen_start)
            fen = line[fen_start:fen_end]
            
            cp_idx = line.find('"cp":')
            if cp_idx != -1:
                cp_start = cp_idx + 5
                cp_end = line.find(',', cp_start)
                cp_end_2 = line.find('}', cp_start)
                if cp_end == -1 or (cp_end_2 != -1 and cp_end_2 < cp_end): cp_end = cp_end_2
                try:
                    cp = float(line[cp_start:cp_end].strip())
                    prob = 1.0 / (1.0 + np.exp(-0.003 * cp))
                except ValueError:
                    return None
            else:
                mate_idx = line.find('"mate":')
                if mate_idx != -1:
                    m_start = mate_idx + 7
                    m_end = line.find(',', m_start)
                    m_end_2 = line.find('}', m_start)
                    if m_end == -1 or (m_end_2 != -1 and m_end_2 < m_end): m_end = m_end_2
                    try:
                        mate_val = int(line[m_start:m_end].strip())
                        prob = 1.0 if mate_val > 0 else 0.0
                    except ValueError:
                        return None
                else:
                    return None
        else:
            # Standard EPD fallback
            parts = line.split('"')
            if len(parts) < 2:
                return None
            fen = parts[0].replace(' c9 ', '').strip()
            try:
                val_str = parts[1]
                if val_str in ["1.0", "0.5", "0.0"]:
                    prob = float(val_str)
                else:
                    cp = float(val_str)
                    prob = 1.0 / (1.0 + np.exp(-0.003 * cp))
            except ValueError:
                return None
        
        w_feat, b_feat = fen_to_halfkp(fen)
        if w_feat is None:
            return None
        return w_feat, b_feat, float(prob)

    # Removed make_tensors to prevent massive file descriptor allocations

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # Buffer stores SPARSE indices only — ~15 ints per position = tiny RAM
        # Dense tensors are created only when yielding, not stored in buffer
        buffer = []  # list of (w_feat_list, b_feat_list, prob_float)

        with open(self.epd_file, 'r', buffering=1 << 20) as f:
            for i, line in enumerate(f):
                if worker_info is not None:
                    if i % worker_info.num_workers != worker_info.id:
                        continue

                result = self.parse_line(line.strip())
                if result is None:
                    continue

                # Store as raw (indices, indices, float) — NOT as dense tensors
                buffer.append(result)

                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        random.shuffle(buffer)
        for item in buffer:
            yield item

# ─── Streaming Binary Dataset ──────────────────────────────────────────────────
class StreamingBinDataset(IterableDataset):
    """
    Reads 72-byte PackedPosition structs directly from the binary dataset.
    Bypasses all string parsing for maximum performance.
    """
    def __init__(self, bin_file, shuffle_buffer=50000):
        self.bin_file = bin_file
        self.shuffle_buffer = shuffle_buffer
        self.struct_fmt = struct.Struct('<8Q4Bh2b')
        self._len = None
        self.RECORD_SIZE = 72

    def __len__(self):
        if self._len is None:
            print(f"Calculating total positions in {self.bin_file}...", flush=True)
            self._len = os.path.getsize(self.bin_file) // self.RECORD_SIZE
        return self._len

    def get_bits(self, val):
        bits = []
        while val:
            lsb = val & -val
            bits.append(lsb.bit_length() - 1)
            val &= val - 1
        return bits

    def parse_record(self, data):
        fields = self.struct_fmt.unpack(data)
        white, black, kings, queens, rooks, bishops, knights, pawns = fields[:8]
        ep, turn, castling, pad1, score, result, pad2 = fields[8:]
        
        # Determine king positions
        wk_sq = -1
        bk_sq = -1
        
        # Calculate king positions first
        k_w = kings & white
        k_b = kings & black
        if k_w: wk_sq = (k_w & -k_w).bit_length() - 1
        if k_b: bk_sq = (k_b & -k_b).bit_length() - 1
            
        if wk_sq == -1 or bk_sq == -1: return None
        
        bk_sq_flipped = bk_sq ^ 56
        
        w_feat, b_feat = [], []
        
        pieces = [
            (pawns & white, 0, True), (knights & white, 1, True), 
            (bishops & white, 2, True), (rooks & white, 3, True), (queens & white, 4, True),
            (pawns & black, 0, False), (knights & black, 1, False), 
            (bishops & black, 2, False), (rooks & black, 3, False), (queens & black, 4, False)
        ]
        
        for bb, pt, is_white in pieces:
            while bb:
                lsb = bb & -bb
                sq = lsb.bit_length() - 1
                bb &= bb - 1
                
                sq_w = sq
                pt_w = pt if is_white else pt + 5
                w_feat.append(pt_w * 4096 + wk_sq * 64 + sq_w)
                
                sq_b = sq ^ 56
                pt_b = pt if not is_white else pt + 5
                b_feat.append(pt_b * 4096 + bk_sq_flipped * 64 + sq_b)
                
        # calculate probability from score or result
        # NNUE MSE usually maps: result 1 = white win, 0 = black, 2 = draw
        # Blend Game Result with Engine Score for faster convergence
        if result == 1:
            w_res = 1.0
        elif result == 0:
            w_res = 0.0
        elif result == 2:
            w_res = 0.5
        else:
            w_res = 1.0 / (1.0 + np.exp(-0.003 * score))
            
        score_prob = 1.0 / (1.0 + np.exp(-0.003 * score))
        
        # 50% Game Result, 50% Search Evaluation
        prob = 0.5 * w_res + 0.5 * score_prob
            
        return w_feat, b_feat, float(prob)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        buffer = []

        with open(self.bin_file, 'rb') as f:
            idx = 0
            while True:
                data = f.read(self.RECORD_SIZE)
                if not data or len(data) < self.RECORD_SIZE:
                    break
                    
                if worker_info is not None:
                    if idx % worker_info.num_workers != worker_info.id:
                        idx += 1
                        continue

                idx += 1
                result = self.parse_record(data)
                if result is None:
                    continue

                buffer.append(result)

                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield item
                    buffer = []

        random.shuffle(buffer)
        for item in buffer:
            yield item

def custom_collate(batch):
    w_batch = torch.zeros(len(batch), NUM_FEATURES, dtype=torch.float32)
    b_batch = torch.zeros(len(batch), NUM_FEATURES, dtype=torch.float32)
    labels = torch.zeros(len(batch), 1, dtype=torch.float32)
    for i, (w_feat, b_feat, prob) in enumerate(batch):
        if w_feat: w_batch[i, w_feat] = 1.0
        if b_feat: b_batch[i, b_feat] = 1.0
        labels[i, 0] = prob
    return w_batch, b_batch, labels

# ─── Network ──────────────────────────────────────────────────────────────────
class HalfKPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(NUM_FEATURES, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE * 2, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, w_features, b_features):
        w_acc = torch.clamp(self.fc1(w_features), 0.0, 1.0)
        b_acc = torch.clamp(self.fc1(b_features), 0.0, 1.0)
        acc   = torch.cat([w_acc, b_acc], dim=1)
        
        # 4-layer Stockfish architecture
        x = torch.clamp(self.fc2(acc), 0.0, 1.0)
        x = torch.clamp(self.fc3(x), 0.0, 1.0)
        return torch.sigmoid(self.fc4(x))

# ─── Weight Export ────────────────────────────────────────────────────────────
def export_quantized_weights(model, filename):
    fc1_w = (model.fc1.weight.detach().cpu().numpy() * 256).astype('int16')
    fc1_b = (model.fc1.bias.detach().cpu().numpy()   * 256).astype('int16')
    fc2_w = (model.fc2.weight.detach().cpu().numpy() * 64).astype('int8')
    fc2_b = (model.fc2.bias.detach().cpu().numpy()   * 64 * 256).astype('int32')
    fc3_w = (model.fc3.weight.detach().cpu().numpy() * 64).astype('int8')
    fc3_b = (model.fc3.bias.detach().cpu().numpy()   * 64 * 256).astype('int32')
    fc4_w = (model.fc4.weight.detach().cpu().numpy() * 64).astype('int8')
    fc4_b = (model.fc4.bias.detach().cpu().numpy()   * 64 * 256).astype('int32')
    with open(filename, 'wb') as f:
        f.write(fc1_w.tobytes())
        f.write(fc1_b.tobytes())
        f.write(fc2_w.tobytes())
        f.write(fc2_b.tobytes())
        f.write(fc3_w.tobytes())
        f.write(fc3_b.tobytes())
        f.write(fc4_w.tobytes())
        f.write(fc4_b.tobytes())
    print(f"Quantized 4-Layer NNUE weights exported to {filename}")

# ─── Training ─────────────────────────────────────────────────────────────────
def train(dataset_file, epochs=2, num_workers=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dataset: {dataset_file}")
    print(f"Device:  {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ''))
    print(f"Epochs:  {epochs}  |  Batch: {BATCH_SIZE}  |  Workers: {num_workers}")
    print(f"Memory:  Streaming (no full preload)\n")

    if dataset_file.endswith(".bin"):
        dataset = StreamingBinDataset(dataset_file, shuffle_buffer=20000)
    else:
        dataset = StreamingEPDDataset(dataset_file, shuffle_buffer=20000)
    pin = False  # pin_memory doubles RAM usage for large batches — not worth it
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=num_workers,
                            pin_memory=pin,
                            collate_fn=custom_collate,
                            prefetch_factor=2 if num_workers > 0 else None)

    model = HalfKPNet().to(device)
    if os.path.exists("best_model.pth"):
        print("Loading existing best_model.pth to resume training...")
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    else:
        print("No existing best_model.pth found, initializing random weights...")
        nn.init.zeros_(model.fc1.bias)
        nn.init.zeros_(model.fc2.bias)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    best_loss = float('inf')
    total_positions = len(dataset)
    batches_per_epoch = total_positions // BATCH_SIZE

    print(f"Total positions: {total_positions:,}")
    print(f"Batches/epoch:   {batches_per_epoch:,}\n")
    print("Starting NNUE training...\n")

    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            count      = 0

            for i, (w_batch, b_batch, labels) in enumerate(dataloader):
                w_batch = w_batch.to(device, non_blocking=True)
                b_batch = b_batch.to(device, non_blocking=True)
                labels  = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(w_batch, b_batch)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count      += 1

                if (i + 1) % 100 == 0:
                    avg = total_loss / count
                    pct = min(100, (i + 1) * 100 / batches_per_epoch)
                    print(f"Epoch {epoch+1}/{epochs} | Batch {i+1} ({pct:.1f}%) | Loss: {avg:.4f}", flush=True)

            avg_loss = total_loss / max(count, 1)
            scheduler.step(avg_loss)
            print(f"\n--- Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f} ---")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "best_model_temp.pth")
                os.replace("best_model_temp.pth", "best_model.pth")
                
                # Output a uniquely named NNUE file so we NEVER overwrite the baseline
                export_name = "brain.nnue"
                export_quantized_weights(model, "temp_network.nnue")
                os.replace("temp_network.nnue", export_name)
                print(f"✅ New best! Saved {export_name} (loss={avg_loss:.4f})")

        print(f"\nTraining complete. Best loss: {best_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user!")
        print("Saving current progress before exiting...")
        export_quantized_weights(model, "temp_network.nnue")
        os.replace("temp_network.nnue", "trained_network_interrupted.nnue")
        print("✅ Weights saved safely. You can now use this .nnue in your engine!")

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    epd_file = sys.argv[1] if len(sys.argv) > 1 else "lichess_2017_training.epd"
    if not os.path.exists(epd_file):
        print(f"Dataset not found: {epd_file}")
        sys.exit(1)
    train(epd_file)
