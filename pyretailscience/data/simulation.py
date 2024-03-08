from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import strictyaml as yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Configuration schema for the config file yaml
config_schema = yaml.Map(
    {
        "stores": yaml.Map(
            {
                "number_of_stores": yaml.Int(),
            }
        ),
        "transactions": yaml.Map(
            {
                "start_date": yaml.Datetime(),
                "end_date": yaml.Datetime(),
                "start_hour": yaml.Int(),
                "end_hour": yaml.Int(),
                "max_products_per_transaction": yaml.Int(),
            }
        ),
        "customers": yaml.Map(
            {
                "starting_number_of_customers": yaml.Int(),
                "churn_probability": yaml.Float(),
                "average_days_between_purchases": yaml.Int(),
                "average_new_customers_per_day": yaml.Int(),
            }
        ),
        "products": yaml.Seq(
            yaml.Map(
                {
                    "category_0_name": yaml.Str(),
                    "category_0_id": yaml.Int(),
                    "subcategories": yaml.Seq(
                        yaml.Map(
                            {
                                "category_1_name": yaml.Str(),
                                "category_1_id": yaml.Int(),
                                "brands": yaml.Seq(
                                    yaml.Map(
                                        {
                                            "brand_name": yaml.Str(),
                                            "brand_id": yaml.Int(),
                                            "products": yaml.Seq(
                                                yaml.Map(
                                                    {
                                                        "product_name": yaml.Str(),
                                                        "product_id": yaml.Int(),
                                                        "unit_price": yaml.Float(),
                                                    }
                                                )
                                            ),
                                        }
                                    )
                                ),
                            }
                        )
                    ),
                }
            )
        ),
    }
)


def _random_time(rnd_generator: np.random.Generator, start_hour, end_hour) -> time:
    """Generate a random time between start_hour and end_hour.

    Args:
        rnd_generator (np.random.Generator): Random number generator
        start_hour (int): Start hour
        end_hour (int): End hour

    Returns:
        time: Random time between start_hour and end_hour
    """
    hour = rnd_generator.integers(start_hour, end_hour)
    minute = rnd_generator.integers(0, 60)
    second = rnd_generator.integers(0, 60)

    return time(hour, minute, second)


@dataclass
class Product:
    """Dataclass for a product.

    Args:
        category_0 (str): Category 0 name
        category_0_id (int): Category 0 ID
        category_1 (str): Category 1 name
        category_1_id (int): Category 1 ID
        brand_name (str): Brand name
        brand_id (int): Brand ID
        product_name (str): Product name
        product_id (int): Product ID
        unit_price (float): Unit price
        quantity_mean (int): Mean quantity purchased
    """

    category_0: str
    category_0_id: int
    category_1: str
    category_1_id: int
    brand_name: str
    brand_id: int
    product_name: str
    product_id: int
    unit_price: float
    quantity_mean: int


class TransactionGenerator:
    """Generates a random transaction for a customer.

    Args:
        rnd_generator (np.random.Generator): Random number generator
        num_stores (int): Number of stores
        max_products_per_transaction (int): Maximum number of products per transaction
        products (list[Product]): List of Product objects
        start_hour (int): Start hour
        end_hour (int): End hour
    """

    def __init__(
        self,
        rnd_generator: np.random.Generator,
        num_stores: int,
        max_products_per_transaction: int,
        products: list[Product],
        start_hour: int,
        end_hour: int,
    ) -> None:
        self.num_stores = num_stores
        self.rnd_generator = rnd_generator
        self.products = products
        self.max_products_per_transaction = max_products_per_transaction
        self.start_hour = start_hour
        self.end_hour = end_hour

    def generate_transaction(self, customer_id: int, simulation_date: date) -> dict:
        """Generate a random transaction for a customer.

        Args:
            customer_id (int): Customer ID
            simulation_date (date): Simulation date

        Returns:
            dict: Transaction details. For example:
            {
                # UUID for the transaction. We'll change this to a sequential integer later
                "transaction_id": "f6c3c7c4-4c6e-4d6b-8f3f-5d0e1a7d6b7d",
                "transaction_datetime": datetime.datetime(2021, 1, 1, 12, 30, 0),
                "customer_id": 1,
                "product_id": 1,
                "product_name": "Product 1",
                "category_0_name": "Category 0",
                "category_0_id": 1,
                "category_1_name": "Category 1",
                "category_1_id": 1,
                "brand_name": "Brand 1",
                "brand_id": 1,
                "unit_price": 10.0,
                "quantity": 2,
                "total_price": 20.0,
                "store_id": 1,
            }
        """
        # Combine transaction_date with a random time
        simulation_datetime = datetime.combine(
            simulation_date,
            _random_time(
                rnd_generator=self.rnd_generator,
                start_hour=self.start_hour,
                end_hour=self.end_hour,
            ),
        )

        store_id = self.rnd_generator.integers(1, self.num_stores)
        products = self.rnd_generator.choice(
            self.products,
            size=self.rnd_generator.integers(1, self.max_products_per_transaction),
        )
        quantity = self.rnd_generator.integers(1, self.max_products_per_transaction)

        # Generate a UUID for now, but we'll change it to a sequential integer later
        transaction_id = str(uuid.uuid4())

        transaction_lines = []
        for product in products:
            quantity = self.rnd_generator.poisson(product.quantity_mean)
            total_price = float(product.unit_price) * quantity

            transaction = {
                "transaction_id": transaction_id,
                "transaction_datetime": simulation_datetime,
                "customer_id": customer_id,
                "product_id": product.product_id,
                "product_name": product.product_name,
                "category_0_name": product.category_0,
                "category_0_id": product.category_0_id,
                "category_1_name": product.category_1,
                "category_1_id": product.category_1_id,
                "brand_name": product.brand_name,
                "brand_id": product.brand_id,
                "unit_price": float(product.unit_price),
                "quantity": quantity,
                "total_price": total_price,
                "store_id": store_id,
            }

            transaction_lines.append(transaction)

        return transaction_lines


class Customer:
    """Simulates a customer's purchasing behavior.

    Attributes:
        has_churned (bool): Whether the customer has churned

    Args:
        rnd_generator (np.random.Generator): Random number generator
        churn_prob (float): Churn probability
        customer_id (int): Customer ID
        transaction_gen (TransactionGenerator): A common transaction generator shared between users.
        period_between_purchases (int): Period between purchases
    """

    has_churned: bool = False

    def __init__(
        self,
        rnd_generator: np.random.Generator,
        churn_prob: float,
        customer_id: int,
        transaction_gen: TransactionGenerator,
        period_between_purchases: int,
    ) -> None:
        self.rnd_generator = rnd_generator
        self.churn_prob = churn_prob
        self.id = customer_id
        self.transaction_gen = transaction_gen
        self.periods_between_purchases = period_between_purchases
        self.transactions = []

        self.time_to_next_purchase = round(
            # Scale the first purchase by a random number to avoid all customers purchasing at roughly the same time
            self.rnd_generator.poisson(period_between_purchases) * self.rnd_generator.random(),
            0,
        )

    def step(self, date: date = None) -> None:
        """Simulate a step (day) for the customer.

        Here we simulate the customer's behavior for a single day. If the customer is due to make a purchase, we
        generate a transaction. We also simulate whether the customer churns.

        Args:
            date (date): Date of the simulation

        Returns:
            None
        """
        if self.has_churned:
            return

        if self.time_to_next_purchase == 0:  # time to buy!
            logger.debug("Customer made a purchase")
            purchase = self.transaction_gen.generate_transaction(self.id, date)
            self.transactions.extend(purchase)

            # Bernoulli trial to see if customer churns
            if self.rnd_generator.binomial(1, self.churn_prob):
                self.has_churned = True
                logger.debug(f"Customer {self.id} churned")
            else:
                self.time_to_next_purchase = self.rnd_generator.poisson(self.periods_between_purchases)
        else:
            self.time_to_next_purchase -= 1


class Simulation:
    """Simulates a retail environment with customers and transactions.

    Args:
        seed (int): Random seed
        config (dict): A dictionary of the settings of the simulation
    """

    def __init__(
        self,
        seed: int,
        config: dict,
    ) -> None:
        self.seed = seed
        self.config = config

        self.rnd_generator = np.random.default_rng(self.seed)

        self.products = self._load_products()

        self.customers = [
            self._create_customer(customer_id=customer_id)
            for customer_id in range(1, self.config["customers"]["starting_number_of_customers"] + 1)
        ]
        self.transactions = []

    @classmethod
    def from_config_file(cls, seed: int, config_file: str) -> Simulation:
        """Create a Simulation from a config file.

        Args:
            seed (int): Random seed
            config_file (str): Path to the config file

        Returns:
            Simulation: A Simulation object
        """
        with open(config_file, "r") as f:
            try:
                config = yaml.load(f.read(), config_schema).data
            except yaml.YAMLError as error:
                logger.exception(error)
                raise error

        return cls(seed=seed, config=config)

    def step(self, date: date) -> None:
        """Simulate a step (day) for the simulation.

        Args:
            date (date): Date of the simulation

        Returns:
            None
        """
        num_new_customers = self.rnd_generator.poisson(self.config["customers"]["average_new_customers_per_day"])
        logger.debug(f"Adding {num_new_customers} new customers")
        self.customers.extend(
            [
                self._create_customer(customer_id=new_customer_id)
                for new_customer_id in range(len(self.customers) + 1, num_new_customers + 1)
            ]
        )
        # Simulate each customer
        for customer in self.customers:
            customer.step(date)

    def run(self) -> None:
        """Run the simulation.

        Returns:
            None
        """
        start_date = self.config["transactions"]["start_date"].date()
        end_date = self.config["transactions"]["end_date"].date()
        days_in_simulation = [start_date + timedelta(n) for n in range((end_date - start_date).days)]
        for sim_day in tqdm(days_in_simulation, desc="Simulating days"):
            self.step(sim_day)

        transactions = []
        for customer in self.customers:
            transactions.extend(customer.transactions)

        # Change transactions UUIDs to sequential integers
        unique_transaction_ids = set([t["transaction_id"] for t in transactions])
        transaction_id_map = {transaction_id: i for i, transaction_id in enumerate(unique_transaction_ids)}
        for transaction in transactions:
            transaction["transaction_id"] = transaction_id_map[transaction["transaction_id"]]

        self.transactions = transactions

    def _create_customer(self, customer_id: int) -> Customer:
        """Create a customer for use in the simulation.

        Args:
            customer_id (int): Customer ID

        Returns:
            Customer: A Customer object
        """
        return Customer(
            rnd_generator=self.rnd_generator,
            churn_prob=self.config["customers"]["churn_probability"],
            customer_id=customer_id,
            period_between_purchases=self.config["customers"]["average_days_between_purchases"],
            transaction_gen=TransactionGenerator(
                rnd_generator=self.rnd_generator,
                num_stores=self.config["stores"]["number_of_stores"],
                max_products_per_transaction=self.config["transactions"]["max_products_per_transaction"],
                products=self.products,
                start_hour=self.config["transactions"]["start_hour"],
                end_hour=self.config["transactions"]["end_hour"],
            ),
        )

    def _load_products(self) -> list[Product]:
        """Load products from the config file.

        Returns:
            list[Product]: List of Product objects
        """
        products = []

        for category_0 in self.config["products"]:
            for category_1 in category_0["subcategories"]:
                for brand in category_1["brands"]:
                    for product in brand["products"]:
                        products.append(
                            Product(
                                category_0=category_0["category_0_name"],
                                category_0_id=int(category_0["category_0_id"]),
                                category_1=category_1["category_1_name"],
                                category_1_id=int(category_1["category_1_id"]),
                                brand_name=brand["brand_name"],
                                brand_id=int(brand["brand_id"]),
                                product_name=product["product_name"],
                                product_id=int(product["product_id"]),
                                unit_price=product["unit_price"],
                                # TODO: Move this to the config file
                                quantity_mean=self.rnd_generator.poisson(self.rnd_generator.integers(1, 3)),
                            )
                        )
        return products

    def save_transactions(self, output_file: str) -> None:
        """Save the transactions to a file in parquet format.

        Args:
            output_file (str): Output file path

        Returns:
            None
        """
        df = pd.DataFrame(self.transactions)
        logger.info(f"Saving {len(df)} transactions to {output_file}")
        df.to_parquet(output_file, index=False)
