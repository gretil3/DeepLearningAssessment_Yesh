"""Train the DCGAN (convolutional generator and discriminator).

Run from the repository root: ``python -m train.train_dcgan``.
"""

import configs
import data
from models.dcgan import DCGANDiscriminator, DCGANGenerator
from train import gan_engine

MODEL_NAME = "dcgan"


def main() -> None:
    """Train the generator and discriminator adversarially on the training split."""
    data.set_seed()
    train_loader, _, _ = data.get_dataloaders()

    generator = DCGANGenerator().to(configs.DEVICE)
    discriminator = DCGANDiscriminator().to(configs.DEVICE)
    gan_engine.fit_gan(
        generator, discriminator, MODEL_NAME, train_loader,
        configs.DCGAN_LATENT_DIM, configs.DCGAN_LR, configs.DCGAN_BETA1, configs.DCGAN_EPOCHS,
    )


if __name__ == "__main__":
    main()
