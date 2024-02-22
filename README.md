![pyretailscience logo](https://raw.githubusercontent.com/Data-Simply/pyretailscience/main/logo.png)

# PyRetailScience

⚡ Democratizing retail data analytics for all retailers ⚡

## 🤔 What is PyRetailScience?

pyretailscience is a Python package designed for performing analytics on retail data. Additionally, the package includes functionality for generating test data to facilitate testing and development.

## Installation

To install pyretailscience, use the following pip command:

```bash
pip install pyretailscience
```

## Quick Start

### Generating Simulated Data

The `pyretailscience` package provides a command-line interface for generating simulated transaction data.

#### Usage
```bash
pyretailscience --config_file=<config_file_path> [--verbose=<True|False>] [--seed=<seed_number>] [output]
```

#### Options and Arguments
- `--config_file=<config_file_path>`: The path to the configuration file for the simulation. This is a required argument.
- `--verbose=<True|False>`: Optional. Set to `True` to see debug messages. Default is `False`.
- `--seed=<seed_number>`: Optional. Seed for the random number generator used in the simulation. If not provided, a random seed will be used.
- `[output]`: Optional. The path where the generated transactions will be saved in parquet format. If not provided, the transactions will be saved in the current directory.

#### Examples
```bash
# Get the default transaction config file
wget https://raw.githubusercontent.com/Data-Simply/pyretailscience/0.1.1/data/default_data_config.yaml
# Generate the data file
pyretailscience --config_file=default_data_config.yaml --seed=123 transactions.parquet
```
This command will generate a file named `transactions.parquet` with the simulated transaction data, using the configuration file at default data configuration file, and a seed of `123` for the random number generator.


# Contributing

We welcome contributions from the community to enhance and improve pyretailscience. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your fork.
5. Open a pull request to the main repository's `main` branch.

Please make sure to follow the existing coding style and provide unit tests for new features.

## Contributors

<a href="https://github.com/Data-Simply/pyretailscience/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Data-Simply/pyretailscience" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## License

This project is licensed under the Elastic License 2.0 - see the [LICENSE](LICENSE) file for details.
