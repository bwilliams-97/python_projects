import argparse
from models import VanillaVAE, ConditionalVAE
import torch
from torchvision import datasets, transforms
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE example")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=784)
    parser.add_argument("--latent-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=int, default=1e-3)
    parser.add_argument("--model-type", type=str, default="vanilla")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-classes", type=int, default=10)

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
    elif args.model_type == "conditional":
        model_kwargs.update({"n_classes": 10})
        model = ConditionalVAE(**model_kwargs)

    # Load MNIST dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **train_loader_kwargs
    )

    # Train model
    train_kwargs = {
        "learning_rate": args.learning_rate,
        "epochs" : args.epochs,
        "device": args.device,
        "model_type": args.model_type,
        "n_classes": args.n_classes
    }
    train(train_loader, model.to(device), **train_kwargs)