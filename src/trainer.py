from collections import deque

import pandas as pd

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .utils import send_to_device


def dcg(scores, labels, k):
    _, indices = scores.topk(k, dim=-1)
    gains = torch.gather(labels, 1, indices)
    discounts = torch.log2(torch.arange(2, 2 + k, device=scores.device).float())
    return (gains / discounts).sum(dim=1)

def ndcg(scores, labels, k):
    dcg_vals = dcg(scores, labels, k)
    ideal_dcg_vals = dcg(labels, labels, k) + 1e-8
    return (dcg_vals / ideal_dcg_vals).mean()


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        epochs=10,
        margin=0.2,
        lr=1e-4,
        weight_decay=0.0,
        folder_name=None,
        device="cuda",
        early_stopping_patience=3,
        moving_avg_window=3,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.margin = margin
        self.lr = lr
        self.weight_decay = weight_decay
        self.folder_name = folder_name
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_dataloader) * self.epochs
        )

        self.history_path = self.folder_name / "history.csv"
        self.best_model_path = self.folder_name / "best_model.pt"
        self.best_val_ndcg = -float('inf')

        self.early_stopping_patience = early_stopping_patience
        self.moving_avg_window = moving_avg_window
        self.val_ndcg_history = deque(maxlen=self.moving_avg_window)
        self.epochs_since_improvement = 0

    def fit(self):
        self.model.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total_ndcg_1 = 0.0
            total_ndcg_5 = 0.0
            total_ndcg_10 = 0.0

            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.epochs} (Train)")
            for step, (batch_image, batch_text, batch_caption) in enumerate(pbar):
                batch_image = send_to_device(batch_image, self.device)
                batch_text = send_to_device(batch_text, self.device)
                batch_caption = send_to_device(batch_caption, self.device)

                B = batch_image["pixel_values"].shape[0]
                K = batch_text["input_ids"].shape[0] // B

                anchor_embed = self.model.get_image_embedding(batch_image)
                pos_embed = self.model.get_text_embedding(batch_caption)
                neg_embed = self.model.get_text_embedding(batch_text)
                D = anchor_embed.shape[1]
                neg_embed = neg_embed.view(B, K, D)

                anchor_embed = F.normalize(anchor_embed, dim=-1)
                pos_embed = F.normalize(pos_embed, dim=-1)
                neg_embed = F.normalize(neg_embed, dim=-1)

                sim_pos = F.cosine_similarity(anchor_embed, pos_embed, dim=-1)
                sim_neg = torch.matmul(neg_embed, anchor_embed.unsqueeze(-1)).squeeze(-1)
                sim_pos_expanded = sim_pos.unsqueeze(1).expand_as(sim_neg)

                loss = F.margin_ranking_loss(
                    sim_pos_expanded.reshape(-1),
                    sim_neg.reshape(-1),
                    torch.ones_like(sim_neg.reshape(-1)),
                    margin=self.margin
                )

                sim_all = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
                rel_all = torch.zeros_like(sim_all)
                rel_all[:, 0] = 1

                ndcg_1 = ndcg(sim_all, rel_all, k=1)
                ndcg_5 = ndcg(sim_all, rel_all, k=5)
                ndcg_10 = ndcg(sim_all, rel_all, k=10)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                total_loss += loss.item()
                total_ndcg_1 += ndcg_1.item()
                total_ndcg_5 += ndcg_5.item()
                total_ndcg_10 += ndcg_10.item()

                avg_loss = total_loss / (step + 1)
                avg_ndcg_1 = total_ndcg_1 / (step + 1)
                avg_ndcg_5 = total_ndcg_5 / (step + 1)
                avg_ndcg_10 = total_ndcg_10 / (step + 1)

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ndcg1=f"{avg_ndcg_1:.4f}",
                    ndcg5=f"{avg_ndcg_5:.4f}",
                    ndcg10=f"{avg_ndcg_10:.4f}"
                )

            val_metrics = self.validate(epoch)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_ndcg@1": avg_ndcg_1,
                "train_ndcg@5": avg_ndcg_5,
                "train_ndcg@10": avg_ndcg_10,
                "val_loss": val_metrics["loss"],
                "val_ndcg@1": val_metrics["ndcg@1"],
                "val_ndcg@5": val_metrics["ndcg@5"],
                "val_ndcg@10": val_metrics["ndcg@10"]
            }

            df = pd.DataFrame([epoch_metrics])
            if self.history_path.exists():
                df.to_csv(self.history_path, mode='a', header=False, index=False)
            else:
                df.to_csv(self.history_path, index=False)

            print(f"[Epoch {epoch}] Saved metrics to {self.history_path}")

            self.val_ndcg_history.append(val_metrics["ndcg@5"])
            moving_avg_ndcg = sum(self.val_ndcg_history) / len(self.val_ndcg_history)

            if moving_avg_ndcg > self.best_val_ndcg:
                self.best_val_ndcg = moving_avg_ndcg
                self.epochs_since_improvement = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"✅ Best model updated (val_ndcg@5={self.best_val_ndcg:.4f}) → {self.best_model_path}")
            else:
                self.epochs_since_improvement += 1
                print(f"⏳ No improvement. ({self.epochs_since_improvement}/{self.early_stopping_patience})")

            if self.epochs_since_improvement >= self.early_stopping_patience:
                print(f"🛑 Early stopping triggered after {epoch} epochs.")
                break

    def validate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_ndcg_1 = 0.0
        total_ndcg_5 = 0.0
        total_ndcg_10 = 0.0

        pbar = tqdm(self.val_dataloader, desc=f"Epoch {epoch}/{self.epochs} (Val)")

        with torch.no_grad():
            for step, (batch_image, batch_text, batch_caption) in enumerate(pbar):
                batch_image = send_to_device(batch_image, self.device)
                batch_text = send_to_device(batch_text, self.device)
                batch_caption = send_to_device(batch_caption, self.device)

                B = batch_image["pixel_values"].shape[0]
                K = batch_text["input_ids"].shape[0] // B

                anchor_embed = self.model.get_image_embedding(batch_image)
                pos_embed = self.model.get_text_embedding(batch_caption)
                neg_embed = self.model.get_text_embedding(batch_text)
                D = anchor_embed.shape[1]
                neg_embed = neg_embed.view(B, K, D)

                anchor_embed = F.normalize(anchor_embed, dim=-1)
                pos_embed = F.normalize(pos_embed, dim=-1)
                neg_embed = F.normalize(neg_embed, dim=-1)

                sim_pos = F.cosine_similarity(anchor_embed, pos_embed, dim=-1)
                sim_neg = torch.matmul(neg_embed, anchor_embed.unsqueeze(-1)).squeeze(-1)
                sim_pos_expanded = sim_pos.unsqueeze(1).expand_as(sim_neg)

                loss = F.margin_ranking_loss(
                    sim_pos_expanded.reshape(-1),
                    sim_neg.reshape(-1),
                    torch.ones_like(sim_neg.reshape(-1)),
                    margin=self.margin
                )

                sim_all = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)
                rel_all = torch.zeros_like(sim_all)
                rel_all[:, 0] = 1

                ndcg_1 = ndcg(sim_all, rel_all, k=1)
                ndcg_5 = ndcg(sim_all, rel_all, k=5)
                ndcg_10 = ndcg(sim_all, rel_all, k=10)

                total_loss += loss.item()
                total_ndcg_1 += ndcg_1.item()
                total_ndcg_5 += ndcg_5.item()
                total_ndcg_10 += ndcg_10.item()

                avg_loss = total_loss / (step + 1)
                avg_ndcg_1 = total_ndcg_1 / (step + 1)
                avg_ndcg_5 = total_ndcg_5 / (step + 1)
                avg_ndcg_10 = total_ndcg_10 / (step + 1)

                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    ndcg1=f"{avg_ndcg_1:.4f}",
                    ndcg5=f"{avg_ndcg_5:.4f}",
                    ndcg10=f"{avg_ndcg_10:.4f}"
                )

        return {
            "loss": avg_loss,
            "ndcg@1": avg_ndcg_1,
            "ndcg@5": avg_ndcg_5,
            "ndcg@10": avg_ndcg_10
        }