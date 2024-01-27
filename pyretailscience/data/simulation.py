import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import strictyaml as yaml
from loguru import logger
from tqdm import tqdm

# Configuration schema for the config file yaml
schema = yaml.Map(
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
                "average_time_between_purchases": yaml.Int(),
                "average_new_customers_per_day": yaml.Int(),
            }
        ),
        "products": yaml.Seq(
            yaml.Map(
                {
                    "category_0": yaml.Str(),
                    "category_0_id": yaml.Int(),
                    "subcategories": yaml.Seq(
                        yaml.Map(
                            {
                                "category_1": yaml.Str(),
                                "category_1_id": yaml.Int(),
                                "brands": yaml.Seq(
                                    yaml.Map(
                                        {
                                            "brand": yaml.Str(),
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


def _random_time(rnd_generator: np.random.Generator, start_hour, end_hour):
    hour = rnd_generator.integers(start_hour, end_hour)
    minute = rnd_generator.integers(0, 60)
    second = rnd_generator.integers(0, 60)

    return time(hour, minute, second)


@dataclass
class Product:
    category_0: str
    category_0_id: int
    category_1: str
    category_1_id: int
    brand: str
    brand_id: int
    product_name: str
    product_id: int
    unit_price: float
    quantity_mean: int


class TransactionGenerator:
    """Generates a random transaction for a customer."""

    def __init__(
        self,
        rnd_generator: np.random.Generator,
        num_stores: int,
        max_products_per_transaction: int,
        products: list[Product],
        start_hour: int,
        end_hour: int,
    ):
        self.num_stores = num_stores
        self.rnd_generator = rnd_generator
        self.products = products
        self.max_products_per_transaction = max_products_per_transaction
        self.start_hour = start_hour
        self.end_hour = end_hour

    def generate_transaction(self, customer_id: int, simulation_date: date) -> dict:
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
            self.products, size=self.rnd_generator.integers(1, self.max_products_per_transaction)
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
                "date": simulation_datetime,
                "customer_id": customer_id,
                "product_id": product.product_id,
                "product_name": product.product_name,
                "category_0": product.category_0,
                "category_0_id": product.category_0_id,
                "category_1": product.category_1,
                "category_1_id": product.category_1_id,
                "brand": product.brand,
                "brand_id": product.brand_id,
                "unit_price": float(product.unit_price),
                "quantity": quantity,
                "total_price": total_price,
                "store_id": store_id,
            }

            transaction_lines.append(transaction)

        return transaction_lines


class Customer:
    has_churned = False

    def __init__(
        self,
        rnd_generator: np.random.Generator,
        churn_prob: float,
        customer_id: int,
        transaction_gen: TransactionGenerator,
        period_between_purchases: int,
    ):
        self.rnd_generator = rnd_generator
        self.churn_prob = churn_prob
        self.id = customer_id
        self.transaction_gen = transaction_gen
        self.periods_between_purchases = period_between_purchases
        self.transactions = []

        self.time_to_next_purchase = self.rnd_generator.poisson(period_between_purchases)

    def step(self, date: date = None) -> None:
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
    def __init__(
        self,
        seed: int,
        config_file: str,
    ):
        self.seed = seed
        with open(config_file, "r") as f:
            try:
                self.config = yaml.load(f.read(), schema).data
            except yaml.YAMLError as error:
                logger.error(error)
                raise error

        self.rnd_generator = np.random.default_rng(self.seed)

        self.products = self._load_products()

        self.customers = [
            self._create_customer(customer_id=customer_id)
            for customer_id in range(1, self.config["customers"]["starting_number_of_customers"] + 1)
        ]
        self.transactions = []

    def step(self, date: date) -> None:
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
        return Customer(
            rnd_generator=self.rnd_generator,
            churn_prob=self.config["customers"]["churn_probability"],
            customer_id=customer_id,
            period_between_purchases=self.config["customers"]["average_time_between_purchases"],
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
        products = []

        for category_0 in self.config["products"]:
            for category_1 in category_0["subcategories"]:
                for brand in category_1["brands"]:
                    for product in brand["products"]:
                        products.append(
                            Product(
                                category_0=category_0["category_0"],
                                category_0_id=int(category_0["category_0_id"]),
                                category_1=category_1["category_1"],
                                category_1_id=int(category_1["category_1_id"]),
                                brand=brand["brand"],
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
        df = pd.DataFrame(self.transactions)
        logger.info(f"Saving {len(df)} transactions to {output_file}")
        df.to_parquet(output_file, index=False)
