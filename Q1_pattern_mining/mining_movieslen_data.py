
import pandas as pd
from ast import literal_eval
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time
import memory_profiler

def statistics(df, encoded_df):
    # Dataset size
    dataset_size = len(df)
    print("Dataset size:", dataset_size)

    # Number of unique users
    num_unique_users = df['user_id'].nunique()
    print("Number of unique users:", num_unique_users)

    # Number of unique movies
    num_unique_movies = len(encoded_df.columns)
    print("Number of unique movies:", num_unique_movies)

    # Sparse data quantification
    # Calculate density and sparsity based on encoded DataFrame
    density = encoded_df.sum().sum() / (len(encoded_df) * len(encoded_df.columns))
    sparsity = 1 - density
    print("Sparsity of the dataset:", sparsity)



if __name__ == "__main__":
    df = pd.read_csv('../data/pattern mining/MoviesLen-100K.csv')

    df['movies'] = df['movies'].apply(literal_eval)
    

    # Convert the transaction data to the format required by TransactionEncoder
    te = TransactionEncoder()
    te_try = te.fit(df['movies']).transform(df['movies'])

    encoded_df = pd.DataFrame(te_try, columns=te.columns_)
    
    statistics(df, encoded_df)

    # Q1 & Q2: Apply Apriori algorithm with max_len set to 3 and different support thresholds
    # min_support_thresholds = [0.1, 0.2, 0.5]  
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
    #     print("Memory usage difference:", abs(mem_diff), "MiB")
    #     print("Execution time:", end_time - start_time, "seconds")
    #     print()

    #     # Mine association rules
    #     # print("Calculating association rules...")
    #     # rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    #     # print("Association Rules:")
    #     # print(rules)
    #     # print()

    # Q3: Apply Apriori algorithm and FP-growth with different support thresholds
    min_support_thresholds = [0.1, 0.2, 0.5]
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