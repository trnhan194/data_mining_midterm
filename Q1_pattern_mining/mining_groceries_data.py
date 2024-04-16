import csv
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time
import memory_profiler

def statistics(df, encoded_df):
    # Dataset size
    dataset_size = len(df)
    print("Dataset size:", dataset_size)

    # Number of unique transactions
    num_unique_transactions = len(encoded_df)
    print("Number of unique transactions:", num_unique_transactions)

    # Number of unique items
    num_unique_items = len(encoded_df.columns)
    print("Number of unique items:", num_unique_items)

    # Sparse data quantification
    # Calculate density and sparsity based on encoded DataFrame
    density = encoded_df.sum().sum() / (num_unique_transactions * num_unique_items)
    sparsity = 1 - density
    print("Sparsity of the dataset:", sparsity)

if __name__ == "__main__":
    with open('../data/pattern mining/Groceries.csv') as f:
        reader_object = csv.reader(f)
        next(reader_object)  # Skip the header row
        data = list(reader_object)

    new_data = []
    for row in data:
        new_data.append([row[0], row[1:]])

    df = pd.DataFrame(new_data, columns=['index', 'itemList'])
    df = df.set_index('index')

    # Convert the 'itemList' column to lists
    df['itemList'] = df['itemList'].apply(lambda x: [i for i in x if pd.notna(i)])

    # Convert the transaction data to the format required by TransactionEncoder
    te = TransactionEncoder()
    te_try = te.fit(df['itemList']).transform(df['itemList'])

    # Create a DataFrame with the encoded transaction data
    encoded_df = pd.DataFrame(te_try, columns=te.columns_)
    
    statistics(df, encoded_df)

    # Q1 & Q2: Apply Apriori algorithm with max_len set to 3 and different support thresholds
    # min_support_thresholds = [0.0045, 0.01, 0.02] 
    # for min_support in min_support_thresholds:
    #     print(f"Calculating frequent itemsets with minimal support of {min_support}...")
    #     start_time = time.time()
    #     mem_usage_start = memory_profiler.memory_usage(max_usage=True)

    #     frequent_itemsets = apriori(encoded_df, min_support=min_support, max_len=3, use_colnames=True)

    #     end_time = time.time()
    #     mem_usage_end = memory_profiler.memory_usage(max_usage=True)

    #     # print("Frequent Itemsets with minimal support of", min_support)
    #     # print(frequent_itemsets)
    #     # print()

    #     mem_diff = mem_usage_end - mem_usage_start
    #     print("Memory usage difference:", mem_diff, "MiB")
    #     print("Execution time:", end_time - start_time, "seconds")
    #     print()

    #     # # Mine association rules
    #     # print("Calculating association rules...")
    #     # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    #     # print("Association Rules:")
    #     # print(rules)
    #     # print()

    # Q3: Apply Apriori algorithm and FP-growth with different support thresholds
    min_support_thresholds = [0.01, 0.02, 0.05]
    for min_support in min_support_thresholds:
        print(f"Calculating frequent itemsets with minimal support of {min_support}...")
        start_time = time.time()

        # Apriori
        print("Apriori:")
        frequent_itemsets_apriori = apriori(encoded_df, min_support=min_support, max_len=3, use_colnames=True)
        print("Execution time for Apriori:", time.time() - start_time, "seconds")
        print()

        # FP-growth
        print("FP-growth:")
        frequent_itemsets_fpgrowth = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
        print("Execution time for FP-growth:", time.time() - start_time, "seconds")
        print()