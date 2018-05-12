import pandas
from global_params import Params
from mlxtend.frequent_patterns import apriori, association_rules

# Association Rule 
# Basket: each video; Item: country
def country_corr(criteria="confidence", num_country=2): 
    video_country = pandas.read_csv(Params.FILE_NAME_COUNTRY, index_col="video_id")
    common_country = apriori(video_country, min_support=0, use_colnames=True)
    # Choose criteria: "confidence" or "lift
    associate_country = association_rules(common_country, metric=criteria, min_threshold=0.0)
    associate_country["basket_size"] = associate_country["antecedants"].apply(lambda x: len(x))
    associate_country["basket_size"] += associate_country["consequents"].apply(lambda x: len(x))
    # Choose subset by num of country in the rule
    result = associate_country[associate_country["basket_size"] == num_country]
    return result.sort_values(by=["basket_size",criteria], ascending=[True,False])

video_country_assoc = country_corr()
print(video_country_assoc[0:10])