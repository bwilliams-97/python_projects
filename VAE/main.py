import argparse
from models import VanillaVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE example")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--input_size", type=int, default=784)
    parser.add_argument("--latent_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=int, default=1e-3)
    parser.add_argument("--model_type", type=str, default="vanilla")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = torch.device("cuda" if args.device=="cuda" else "cpu")

    model_kwargs = {
        "input_size": args.input_size,
        "latent_size" : args.latent_size
    }

    if args.model_type == "vanilla":
        model = VanillaVAE(**model_kwargs)

    train_kwargs = {
        "learning_rate": args.learning_rate,
        "epochs" : args.epochs
    }

    train(model.to(device), **train_kwargs)