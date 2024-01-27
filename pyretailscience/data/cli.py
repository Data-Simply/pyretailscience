import sys

import click
from loguru import logger

from pyretailscience.data import simulation


@click.command()
@click.option("--config_file", type=click.Path(dir_okay=False))
@click.option("--verbose", type=bool, default=False)
@click.option("--seed", default=1234, type=int)
@click.argument("output_file", type=click.Path(dir_okay=False))
def generate(
    config_file: str,
    verbose: bool,
    seed: int,
    output_file: str,
):
    """Generate a CSV file with random transaction data.

    args:
        config_file (str): Configuration file for the simulation
        verbose (bool): Whether to print debug messages
        seed (int): random seed
        output_file (str): File to write the transactions to in parquet format
    """

    # Set logging level to info
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    logger.info("Generating data...")

    sim = simulation.Simulation(seed=seed, config_file=config_file)
    sim.run()
    sim.save_transactions(output_file)

    logger.info("Done!")


if __name__ == "__main__":
    generate()
