import argparse
from models import VanillaVAE
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE example")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--input_size", type=int, default=784)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--model_type", type=str, default="vanilla")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    # Housekeeping
    torch.manual_seed(0)
    device = torch.device("cuda" if args.device=="cuda" else "cpu")
    train_loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.device=="cuda" else {}

    # Initalise VAE model
    model_kwargs = {
        "input_size": args.input_size,
        "latent_size" : args.latent_size
    }

    if args.model_type == "vanilla":
        model = VanillaVAE(**model_kwargs)

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **train_loader_kwargs
    )

    # Train model
    train_kwargs = {
        "learning_rate": args.learning_rate,
        "epochs" : args.epochs
    }
    train(train_loader, model.to(device), **train_kwargs)

    # Generate sample images
    with torch.no_grad():
        sample = torch.randn(32, args.latent_size).to(device)
        decoded_sample = model.decode(sample).cpu()
        save_image(decoded_sample.view(32, 1, 28, 28),
                    'results/image.png')