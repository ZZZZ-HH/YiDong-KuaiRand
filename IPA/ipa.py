import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm


class RewardModel(nn.Module):
    """
    Session-wise reward model that predicts multiple engagement metrics
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)

        # Towers for different engagement metrics
        self.tower_swt = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.tower_vtr = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.tower_wtr = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.tower_ltr = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, user_emb: torch.Tensor, session_embs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_emb: [batch_size, user_dim]
            session_embs: [batch_size, session_len, item_dim]
        Returns:
            Dictionary of reward scores for each metric
        """
        # Target-aware representation
        target_aware = user_emb.unsqueeze(1) * session_embs  # [batch, session_len, item_dim]

        # Self-attention over session items
        attn_output, _ = self.self_attention(
            target_aware, target_aware, target_aware
        )
        session_rep = torch.sum(attn_output, dim=1)  # [batch, item_dim]

        # Predict all rewards
        return {
            "swt": self.tower_swt(session_rep),
            "vtr": self.tower_vtr(session_rep),
            "wtr": self.tower_wtr(session_rep),
            "ltr": self.tower_ltr(session_rep)
        }


class OneRecWithIPA:
    """
    Main class implementing OneRec with Iterative Preference Alignment
    """

    def __init__(self, model, reward_model, tokenizer, device="cuda"):
        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate_candidates(self, user_history: List[str], num_candidates: int = 128) -> List[Tuple[str, float]]:
        """
        Generate candidate sessions using beam search
        Args:
            user_history: List of historical item IDs
            num_candidates: Number of candidates to generate
        Returns:
            List of (generated_session, reward_score) tuples
        """
        self.model.eval()

        # Tokenize input
        input_ids = self.tokenizer(user_history, return_tensors="pt").to(self.device)

        # Generate candidates with beam search
        outputs = self.model.generate(
            input_ids,
            max_length=50,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            early_stopping=True
        )

        # Decode generated sequences
        candidates = [self.tokenizer.decode(seq, skip_special_tokens=True)
                      for seq in outputs]

        # Score candidates with reward model
        with torch.no_grad():
            user_emb = self.model.get_user_embedding(input_ids)
            session_embs = [self.model.get_session_embedding(cand) for cand in candidates]
            session_embs = torch.stack(session_embs)

            rewards = self.reward_model(user_emb.expand_as(session_embs), session_embs)
            total_rewards = rewards["swt"] + rewards["ltr"]  # Combine watch time and like metrics

        return list(zip(candidates, total_rewards.cpu().numpy().flatten()))

    def create_preference_pairs(self, user_histories: List[List[str]], dpo_sample_ratio: float = 0.01) -> List[
        Tuple[str, str, List[str]]]:
        """
        Create preference pairs for DPO training
        Args:
            user_histories: List of user histories
            dpo_sample_ratio: Fraction of data to use for DPO
        Returns:
            List of (winner_session, loser_session, user_history) tuples
        """
        preference_pairs = []

        # Sample subset of users for DPO
        num_samples = int(len(user_histories) * dpo_sample_ratio)
        sampled_users = np.random.choice(len(user_histories), num_samples, replace=False)

        for idx in tqdm(sampled_users, desc="Creating preference pairs"):
            history = user_histories[idx]
            candidates = self.generate_candidates(history)

            if len(candidates) < 2:
                continue

            # Sort by reward score
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Take top as winner, bottom as loser
            winner = candidates[0][0]
            loser = candidates[-1][0]

            preference_pairs.append((winner, loser, history))

        return preference_pairs

    def dpo_loss(self, winner_ids: torch.Tensor, loser_ids: torch.Tensor, history_ids: torch.Tensor,
                 beta: float = 0.1) -> torch.Tensor:
        """
        Compute DPO loss
        Args:
            winner_ids: Tokenized winner sessions [batch, seq_len]
            loser_ids: Tokenized loser sessions [batch, seq_len]
            history_ids: Tokenized user histories [batch, seq_len]
            beta: Temperature parameter
        Returns:
            DPO loss value
        """
        # Get log probabilities from current model
        winner_logp = self.model(history_ids, winner_ids).log_prob(winner_ids)
        loser_logp = self.model(history_ids, loser_ids).log_prob(loser_ids)

        # Get log probabilities from reference model (detached)
        with torch.no_grad():
            ref_winner_logp = self.model(history_ids, winner_ids).log_prob(winner_ids)
            ref_loser_logp = self.model(history_ids, loser_ids).log_prob(loser_ids)

        # Compute log ratios
        log_ratio_winner = winner_logp - ref_winner_logp
        log_ratio_loser = loser_logp - ref_loser_logp

        # DPO loss
        losses = -torch.log(torch.sigmoid(beta * (log_ratio_winner - log_ratio_loser)))
        return losses.mean()

    def train_ipa(self, user_histories: List[List[str]], num_iterations: int = 3, batch_size: int = 32):
        """
        Perform iterative preference alignment
        Args:
            user_histories: List of user histories
            num_iterations: Number of IPA iterations
            batch_size: Training batch size
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)

        for iteration in range(num_iterations):
            print(f"\n=== IPA Iteration {iteration + 1}/{num_iterations} ===")

            # 1. Create preference pairs
            preference_pairs = self.create_preference_pairs(user_histories)

            if not preference_pairs:
                print("No preference pairs generated - skipping iteration")
                continue

            # 2. Create dataset and dataloader
            dataset = PreferenceDataset(preference_pairs, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 3. Train with DPO loss
            self.model.train()
            total_loss = 0

            for batch in tqdm(dataloader, desc="Training"):
                history_ids = batch["history_ids"].to(self.device)
                winner_ids = batch["winner_ids"].to(self.device)
                loser_ids = batch["loser_ids"].to(self.device)

                optimizer.zero_grad()
                loss = self.dpo_loss(winner_ids, loser_ids, history_ids)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"DPO Loss: {avg_loss:.4f}")


class PreferenceDataset(Dataset):
    """Dataset for storing preference pairs"""

    def __init__(self, preference_pairs: List[Tuple[str, str, List[str]]], tokenizer):
        self.preference_pairs = preference_pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.preference_pairs)

    def __getitem__(self, idx):
        winner, loser, history = self.preference_pairs[idx]

        return {
            "history_ids": self.tokenizer(history, return_tensors="pt").input_ids.squeeze(0),
            "winner_ids": self.tokenizer(winner, return_tensors="pt").input_ids.squeeze(0),
            "loser_ids": self.tokenizer(loser, return_tensors="pt").input_ids.squeeze(0)
        }