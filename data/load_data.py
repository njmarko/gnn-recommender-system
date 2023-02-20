import numpy as np
import pandas as pd
import torch
from collections import defaultdict

pd.set_option('display.max_columns', None)


def sort_cities(population):
    if population < 50000:
        return 'rural'
    if population < 100_000:
        return 'small'
    if population < 250_000:
        return 'small_medium'
    if population < 500_000:
        return 'medium'
    if population < 1_000_000:
        return 'medium_large'
    return 'large'


def read_customers():
    users = pd.read_csv('ecommerce/olist_customers_dataset.csv')
    st_inhabitants = pd.read_csv('ecommerce/br_state_inhabitants.csv')
    st_codes = pd.read_csv('ecommerce/br_state_codes.csv')
    st_grp = pd.read_csv('ecommerce/br_state_grp.csv')  # gross regional product
    ct_inhabitants = pd.read_csv('ecommerce/br_cities_population.csv')

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

    return users


def read_product_translations():
    translations = pd.read_csv('ecommerce/product_category_name_translation.csv')
    return translations


def get_product_translation(product, translation_df):
    x = translation_df[translation_df['product_category_name'] == product]['product_category_name_english'].iloc[0]
    return x


def read_products(translate=False):
    products = pd.read_csv('ecommerce/olist_products_dataset.csv')

    if translate:
        product_name_translations = read_product_translations()
        products['product_category_name'] = products['product_category_name'].apply(
            lambda product: get_product_translation(product, product_name_translations)
        )

    return products


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


def create_graph_edges():
    orders = pd.read_csv('ecommerce/olist_orders_dataset.csv')
    order_items = pd.read_csv('ecommerce/olist_order_items_dataset.csv')
    users = pd.read_csv('ecommerce/olist_customers_dataset.csv')
    ratings = pd.read_csv('ecommerce/olist_order_reviews_dataset.csv')
    edge_dict = defaultdict(dict)

    # Ne bi trebalo da moze da vise review-ova ima isti review_id
    # Ne bi trebalo da moze da vise review-ova ocenjuju istu narudzbinu
    # Medjutim to je slucaj
    # Kreatori data seta kazu da je greska, medjutim nisu popravili???
    ratings.drop_duplicates(subset=['review_id'])
    ratings.drop_duplicates(subset=['order_id'])

    for row in orders.itertuples(index=False):
        customer_id, order_id = row.customer_id, row.order_id
        user_id = users[users['customer_id' == customer_id]]['unique_customer_id']
        user_items = order_items[order_items['order_id' == order_id]]
        for item in user_items.itertuples(index=False):
            if item.order_item_id > 1:
                print('stop me')
            try:
                edge_dict[user_id][item.product_id] += 1
            except KeyError:
                edge_dict[user_id][item.product_id] = Edge(user_id, item.product_id)
            if not ratings[ratings['order_id'] == item.order_id].empty:
                rating = ratings[ratings['order_id'] == item.order_id].iloc[0]['review_score']
                edge_dict[user_id][item.product_id].set_rating(rating)

    return edge_dict




if __name__ == '__main__':
    e = Edge('1', '1')
    print(e.product_purchase_count)
    e += 1
    print(e.product_purchase_count)