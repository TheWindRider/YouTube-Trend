import json
import pandas

# Global Params
data_folder = "data/"
country_list = ["CA", "DE", "FR", "GB", "US"]

# Parse JSON File
country = []
category_id = []
category_name = []
for each_country in country_list: 
    each_file = data_folder + each_country + "_category_id.json"
    each_lookup = json.load(open(each_file, 'r'))
    for each_category in each_lookup["items"]: 
        country.append(each_country)
        category_id.append(each_category["id"])
        category_name.append(each_category["snippet"]["title"])

# Build Table
category_lkp = pandas.DataFrame.from_dict({"country_code": country, "id": category_id, "name": category_name})
category_lkp.to_csv("category_lkp.csv", index=False)
