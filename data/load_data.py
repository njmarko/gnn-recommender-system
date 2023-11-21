import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

pd.set_option('display.max_columns', None)


def sort_cities(population):
    if population < 50000:
        return 0  # 'rural'
    if population < 100_000:
        return 1  # 'small'
    if population < 250_000:
        return 2  # 'small_medium'
    if population < 500_000:
        return 3  # 'medium'
    if population < 1_000_000:
        return 4  # 'medium_large'
    return 5  # 'large'


def read_customers() -> tuple[pd.DataFrame, dict]:
    users = pd.read_csv('data/ecommerce/olist_customers_dataset.csv')
    st_inhabitants = pd.read_csv('data/ecommerce/br_state_inhabitants.csv')
    st_codes = pd.read_csv('data/ecommerce/br_state_codes.csv')
    st_grp = pd.read_csv('data/ecommerce/br_state_grp.csv')  # gross regional product
    ct_inhabitants = pd.read_csv('data/ecommerce/br_cities_population.csv')

    st_inhabitants['population'] /= 1000    # convert to milli vanillions
    st_inhabitants = pd.merge(
        st_inhabitants,
        st_codes[['subdivision', 'name']],
        how='left', left_on='state', right_on='name'
    )[['subdivision', 'state', 'population']]

    st_info = pd.merge(st_inhabitants, st_grp, how='right', on='state')

    st_info['grp_per_capita'] = st_info['grp'] / st_info['population']

    users = pd.merge(users, st_info, how='left', left_on='customer_state', right_on='subdivision')
    users.drop(columns=['subdivision', 'state', 'grp'], inplace=True)
    users['grp_per_capita'] *= 190  # convert to dollars (* 1000 * 0.19)

    ct_inhabitants['city'] = ct_inhabitants['city'].str.lower()
    ct_population = ct_inhabitants[['city', 'pop2023']]

    users = pd.merge(users, ct_population, how='left', left_on='customer_city', right_on='city').drop(columns='city')
    users['pop2023'].fillna(0, inplace=True)
    users['living_category'] = users['pop2023'].apply(lambda row: sort_cities(row))

    le = LabelEncoder()
    le.fit(users['customer_state'])

    users['customer_state_code'] = le.transform(users['customer_state'])
    # TODO: See if pop2023 should be dropped as well
    users.drop(columns=['customer_state', 'customer_city', 'customer_zip_code_prefix', 'customer_id'], inplace=True)
    users.set_index('customer_unique_id', inplace=True)

    scaler = StandardScaler()
    columns_to_scale = ['population', 'grp_per_capita', 'pop2023']
    users[columns_to_scale] = scaler.fit_transform(users[columns_to_scale])

    mapping = {index: i for i, index in enumerate(users.index.unique())}

    return users, mapping


def read_product_translations():
    translations = pd.read_csv('data/ecommerce/product_category_name_translation.csv')
    return translations


def get_product_translation(product, translation_df):
    x = translation_df[translation_df['product_category_name'] == product]['product_category_name_english'].iloc[0]
    return x


def read_products(translate=False) -> tuple[pd.DataFrame, dict]:
    products = pd.read_csv('data/ecommerce/olist_products_dataset.csv', index_col='product_id')
    products['product_category_name'].fillna('N/A', inplace=True)
    products.fillna(0, inplace=True)

    product_purchase_prices = pd.read_csv('data/ecommerce/olist_order_items_dataset.csv')[['product_id', 'price']]
    product_purchase_prices = product_purchase_prices.groupby('product_id').mean()
    products = pd.merge(products, product_purchase_prices, how='left', on='product_id')

    products['price'].fillna(products['price'].mean())
    print(products.head())

    le = LabelEncoder()
    le.fit(products['product_category_name'])
    products['product_category_code'] = le.transform(products['product_category_name'])

    if translate:
        product_name_translations = read_product_translations()
        products['product_category_name'] = products['product_category_name'].apply(
            lambda product: get_product_translation(product, product_name_translations)
        )

    products.drop(columns='product_category_name', inplace=True)

    scaler = StandardScaler()
    columns_to_scale = [
        'product_name_length', 'product_description_length',
        'product_weight_g',
        'product_length_cm', 'product_width_cm', 'product_height_cm'
    ]
    products[columns_to_scale] = scaler.fit_transform(products[columns_to_scale])

    mapping = {index: i for i, index in enumerate(products.index.unique())}

    return products, mapping


class Edge:
    def __init__(self, customer_id, product_id):
        self.customer_id = customer_id
        self.product_id = product_id
        self.rating = np.NaN
        self.product_purchase_count = 1

    def set_rating(self, rating):
        self.rating = rating

    def __add__(self, other):
        self.product_purchase_count += other
        return self

    def to_tensor(self):
        return torch.Tensor([self.rating, self.product_purchase_count])



def convert_to_unix_timestamp(timestamp_str, format='%Y-%m-%d %H:%M:%S'):
    # Convert string to datetime object
    dt_object = datetime.strptime(timestamp_str, format)

    # Convert datetime object to Unix timestamp
    unix_timestamp = int(dt_object.timestamp())

    return unix_timestamp

def create_graph_edges() -> pd.DataFrame:
    orders = pd.read_csv('data/ecommerce/olist_orders_dataset.csv')
    order_items = pd.read_csv('data/ecommerce/olist_order_items_dataset.csv')
    users = pd.read_csv('data/ecommerce/olist_customers_dataset.csv')
    ratings = pd.read_csv('data/ecommerce/olist_order_reviews_dataset.csv')

    # Ne bi trebalo da moze da vise review-ova ima isti review_id
    # Ne bi trebalo da moze da vise review-ova ocenjuju istu narudzbinu
    # Medjutim to je slucaj
    # Kreatori data seta kazu da je greska, medjutim nisu popravili???
    ratings.drop_duplicates(subset=['review_id'])
    ratings.drop_duplicates(subset=['order_id'])

    user_orders = pd.merge(users, orders, how='inner', on='customer_id')
    user_orders_reviewed = pd.merge(user_orders, ratings, how='left', on='order_id')
    user_items = pd.merge(user_orders_reviewed, order_items, how='right', on='order_id')

    user_item_counts = (
        user_items
        .groupby(['customer_unique_id', 'product_id'])
        .size()
        .reset_index().rename(columns={0: 'purchase_count'})
    )

    # print(user_item_counts[user_item_counts['customer_unique_id'] == 'bb8a37225e0279ac8a274c9765617eaf'])

    user_items = (
        pd
        .merge(user_items, user_item_counts, how='outer', on=['customer_unique_id', 'product_id'])
        .drop_duplicates(subset=['customer_unique_id', 'product_id'])
    )

    user_items['review_score'].fillna(value=3, inplace=True)

    # print(user_items[['customer_unique_id', 'product_id', 'review_score']]
    #       [user_items['order_id'] == '005d9a5423d47281ac463a968b3936fb' ])  # '001ab0a7578dd66cd4b0a71f5b6e1e41'])

    user_items['timestamp'] = pd.to_datetime(user_items['order_purchase_timestamp'])
    user_items['timestamp'] = pd.to_numeric(user_items['timestamp']) // 10**9

    return user_items[['customer_unique_id', 'product_id', 'review_score', 'purchase_count', 'timestamp']]


if __name__ == '__main__':
    _customers, customer_mapping = read_customers()
    _products, product_mapping = read_products()

    # print(_customers.columns)
    # print(_products.columns)
    # print(_products.columns)
    #
    # print(_customers.head().values)
    # print(_products.head().values)
