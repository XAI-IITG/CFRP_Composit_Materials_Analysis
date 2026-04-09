import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================================
# 3.2 TRANSFORMER and LSTM MODEL FOR RUL PREDICTION
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerRULPredictor(nn.Module):
    """Transformer model for RUL prediction"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1):
        super(TransformerRULPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model dimensions
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, d_model)
        
        # Final prediction layers
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        
        return x.squeeze(-1)

# ============================================================================
# 3.3 LSTM MODEL FOR RUL PREDICTION
# ============================================================================

class LSTMRULPredictor(nn.Module):
    """LSTM model for RUL prediction"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super(LSTMRULPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Final prediction layers
        x = self.dropout(last_output)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output in [0, 1] range
        
        return x.squeeze(-1)

# ============================================================================
# 3.4 KG-ENRICHED TRANSFORMER FOR RUL PREDICTION
# ============================================================================

class KGEnrichedTransformerRULPredictor(nn.Module):
    """
    Transformer model enriched with Knowledge Graph embeddings.
    
    KG embeddings are projected to d_model dimensions and added to the
    time-series token embeddings after input projection + positional encoding.
    This provides free structural context (layup type, damage progression)
    without changing the sequence structure.
    """
    
    def __init__(self, input_dim, kg_embed_dim=64, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super(KGEnrichedTransformerRULPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.kg_embed_dim = kg_embed_dim
        
        # Input projection (time-series features -> d_model)
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # KG embedding projection (kg_embed_dim -> d_model)
        self.kg_projection = nn.Sequential(
            nn.Linear(kg_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Trying concatenation : Fusion layer after concatenation
        ###
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(2 * d_model, d_model),
        #     nn.LayerNorm(d_model),
        #     nn.ReLU(),
        #     nn.Dropout(dropout)
        # )

        ####

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x, kg_embed):
        """
        Args:
            x: Time-series input (batch_size, seq_len, input_dim)
            kg_embed: KG embeddings (batch_size, kg_embed_dim)
        
        Returns:
            RUL predictions (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project time-series input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Project KG embedding to d_model
        kg_ctx = self.kg_projection(kg_embed)  # (batch_size, d_model)
        

        #############
        ### ----- Concatenation method
        # NEW: Expand KG to match sequence length
        # kg_ctx = kg_ctx.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, d_model)
        
        # # NEW: Concatenate instead of adding
        # x = torch.cat([x, kg_ctx], dim=-1)  # (batch, seq_len, 2*d_model)
        
        # # NEW: Project back to d_model
        # x = self.fusion_layer(x)  # (batch, seq_len, d_model)
        ############

        ### ----- Addition method
        # Add KG context to every timestep (broadcast)
        x = x + kg_ctx.unsqueeze(1)  # (batch_size, seq_len, d_model)
        ############

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Apply transformer
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Final prediction layers
        x = self.dropout(pooled)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)