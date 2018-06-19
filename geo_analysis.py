import pandas
from mlxtend.frequent_patterns import apriori, association_rules
from global_params import Params

def pivot_country(): 
    file_names = [Params.DIR_DATA + country + "videos.csv" for country in Params.COUNTRY_ALL]
    videos_all = [pandas.read_csv(video_file, usecols=["video_id"]) for video_file in file_names]
    country = []
    videos = []
    # Extract and keep unique IDs
    for i in range(len(Params.COUNTRY_ALL)): 
        country_curr = Params.COUNTRY_ALL[i]
        video_curr = set(videos_all[i]["video_id"])
        print("{}: {} videos, {} unique".format(country_curr, len(videos_all[i]), len(video_curr)))
        # Extend to the end of final list
        country.extend([country_curr for i in range(len(video_curr))])
        videos.extend(list(video_curr))
    # Pivot by countries
    video_country = pandas.DataFrame.from_dict({"video_id": videos, "country": country})[["video_id", "country"]]
    video_country = video_country.pivot_table(index="video_id", columns="country", aggfunc=len, fill_value=0.0)
    # Put index as a field before save to file
    video_country = pandas.DataFrame(video_country.to_records())
    video_country.to_csv(Params.DIR_DATA + Params.FILE_NAME_COUNTRY, index=False)

def country_corr(criteria="confidence", num_country=2): 
    """ Association Rule
    Basket: each video; Item: country
    This tries to reveal geographical correlation about video trending
    """
    video_country = pandas.read_csv(Params.DIR_DATA + Params.FILE_NAME_COUNTRY, index_col="video_id")
    common_country = apriori(video_country, min_support=0, use_colnames=True)
    # Choose criteria: "confidence" or "lift" are quite common
    associate_country = association_rules(common_country, metric=criteria, min_threshold=0.0)
    # Basket size is the sum of "antecedants" and "consequents" sizes
    associate_country["basket_size"] = associate_country["antecedants"].apply(lambda x: len(x))
    associate_country["basket_size"] += associate_country["consequents"].apply(lambda x: len(x))
    # Choose subset by num of country in the rule
    result = associate_country[associate_country["basket_size"] == num_country]
    return result.sort_values(by=criteria, ascending=False)
    
