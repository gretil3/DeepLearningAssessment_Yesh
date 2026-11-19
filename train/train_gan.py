"""Train the Vanilla GAN (MLP generator and discriminator).

Run from the repository root: ``python -m train.train_gan``.
"""

import configs
import data
from models.gan import Discriminator, Generator
from train import gan_engine

MODEL_NAME = "vanilla_gan"


def main() -> None:
    """Train the generator and discriminator adversarially on the training split."""
    data.set_seed()
    train_loader, _, _ = data.get_dataloaders()

    generator = Generator().to(configs.DEVICE)
    discriminator = Discriminator().to(configs.DEVICE)
    gan_engine.fit_gan(
        generator, discriminator, MODEL_NAME, train_loader,
        configs.GAN_LATENT_DIM, configs.GAN_LR, configs.GAN_BETA1, configs.GAN_EPOCHS,
    )


if __name__ == "__main__":
    main()
