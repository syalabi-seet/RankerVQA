import uuid
import json

from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from src.loader import ImageCaptionDataset
from src.model import ImageCaptionModel
from src.trainer import Trainer
from src.utils import collate_fn


def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train_n_samples", type=int, default=100)
    parser.add_argument("--val_n_samples", type=int, default=500)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--k_neg", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--proj_depth", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--text_model_name", 
        type=str, 
        default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--image_model_name", 
        type=str, 
        default="facebook/deit-small-distilled-patch16-224"
    )

    args = parser.parse_args()
    
    folder_name = Path("experiments") / str(uuid.uuid1())
    folder_name.mkdir(parents=True, exist_ok=True)
    
    with open(folder_name / "config.json", "w") as f:
        json.dump(vars(args), f, indent=4)                

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create dataloader
    train_dataset = ImageCaptionDataset(
        split="train", 
        n_samples=args.train_n_samples,
        k=args.k_neg
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    val_dataset = ImageCaptionDataset(
        split="val", 
        n_samples=args.val_n_samples,
        k=args.k_neg
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )

    model = ImageCaptionModel(
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        proj_depth=args.proj_depth,
        dropout=args.dropout,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha        
    )
    model.to(device)

    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        margin=args.margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        folder_name=folder_name,
        device=device
    )
    trainer.fit()


if __name__ == "__main__":
    main()