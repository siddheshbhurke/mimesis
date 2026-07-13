import torch
from pathlib import Path 
import argparse
from utils.utils import ImageFolderDataset, get_transform, adaptive_instance_normalization, calc_mean_std
from torch.utils.data import DataLoader
from utils.models import VGGEncoder, Decoder
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type = str, default = r'C:\Users\Siddhesh\Desktop\nst-project\content_data_sample',
                        help = "Location of content dataset")

    parser.add_argument('--style_dir', type = str, default = r'C:\Users\Siddhesh\Desktop\nst-project\style_data_sample',
                        help = "Location of style dataset")
    
    parser.add_argument('--vgg', type = str, default = r'C:\Users\Siddhesh\Desktop\nst-project\vgg_normalised.pth',
                        help = "Location of pre-trained vgg")
    
    parser.add_argument('--experiment', type = str, default = r'experiment',
                        help = "Name of experiment")
    
    parser.add_argument('--final_size', type = int, default = 512,
                           help = "Size of final image")
    
    parser.add_argument('--content_size', type = int, default = 256,
                           help = "Size of content image")
    
    parser.add_argument('--style_size', type = int, default = 256,
                           help = "Size of content image")
    
    parser.add_argument('--crop', action = 'store_true' , default = True,
                           help = "Crop Image")

    parser.add_argument('--batch_size', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-4,
                        help = 'Learning Rate')

    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')
    
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')

    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    
    parser.add_argument('--style_weight', type=float, default=10,
                        help='Style weight')
    
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')
    
    parser.add_argument('--save_interval', type=int, default=2,
                        help='Save interval')
    
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')
    
    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')
    
    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')
    
    parser.add_argument(
    '--scheduler_path',
    type=str,
    default=None,
    help='Path to scheduler checkpoint'
)
   
    return parser.parse_args()



def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path("experiment") / args.experiment
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(save_dir / "args.txt", "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    # -------------------- Datasets --------------------
    content_transform = get_transform(
        args.content_size,
        args.crop,
        args.final_size
    )

    style_transform = get_transform(
        args.style_size,
        args.crop,
        args.final_size
    )

    content_dataset = ImageFolderDataset(
        args.content_dir,
        content_transform
    )

    style_dataset = ImageFolderDataset(
        args.style_dir,
        style_transform
    )

    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    if len(content_dataloader) == 0 or len(style_dataloader) == 0:
        raise ValueError(
            "Dataset is too small for the selected batch size."
        )

    # -------------------- Models --------------------
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    encoder.eval()
    decoder.train()

    mse_loss = torch.nn.MSELoss()

    optimizer = optim.Adam(
        decoder.parameters(),
        lr=args.lr
    )

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        decoder.load_state_dict(
            torch.load(args.decoder_path, map_location=device)
        )

        optimizer.load_state_dict(
            torch.load(args.optimizer_path, map_location=device)
        )

        scheduler.load_state_dict(
        torch.load(args.scheduler_path, map_location=device)
    )
    print("Training...")

    # -------------------- Training --------------------
    for epoch in range(args.epochs):

        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0

        num_batches = min(
            len(content_dataloader),
            len(style_dataloader)
        )

        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=num_batches,
            desc=f"Epoch {epoch+1}/{args.epochs}"
        )

        for content_batch, style_batch in progress_bar:

            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            # Encoder features
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # AdaIN
            t = adaptive_instance_normalization(
                c_feats[-1],
                s_feats[-1]
            )

            # Decoder
            g = decoder(t)

            # Re-encode generated image
            g_feats = encoder(g)

            # Content loss
            loss_c = (
                mse_loss(g_feats[-1], t)
                * args.content_weight
            )

            # Style loss
            loss_s = 0

            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)

                loss_s += (
                    mse_loss(g_mean, s_mean)
                    + mse_loss(g_std, s_std)
                )

            loss_s *= args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                decoder.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}",
                Content=f"{loss_c.item():.4f}",
                Style=f"{loss_s.item():.4f}"
            )

        scheduler.step()

        running_loss /= num_batches
        running_closs /= num_batches
        running_sloss /= num_batches

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f"Epoch {epoch+1}: "
                f"Loss={running_loss:.4f} | "
                f"Content={running_closs:.4f} | "
                f"Style={running_sloss:.4f}"
            )

        if (epoch + 1) % args.save_interval == 0:

            checkpoint = {
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            torch.save(checkpoint, save_dir / "checkpoint.pth")

            with torch.no_grad():
                output = torch.cat(
                    [content_batch, style_batch, g],
                    dim=0
                )

                save_image(
                    output,
                    save_dir / f"output_{epoch+1}.png",
                    nrow=args.batch_size,
                    normalize=True
                )


if __name__ == "__main__":
    main()