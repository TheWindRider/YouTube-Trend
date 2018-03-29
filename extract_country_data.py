import pandas

# Global Params
data_folder = "data/"
country_list = ["CA", "DE", "FR", "GB", "US"]

videoID_list = [pandas.read_csv(data_folder + each_country + "videos.csv", usecols=["video_id"]) 
                for each_country in country_list]
country = []
videos = []

# Extract, Keep Unique
for i in range(len(country_list)): 
    countryA = country_list[i]
    videoA = set(videoID_list[i]["video_id"])
    print("{}: {} videos, {} unique".format(countryA, len(videoID_list[i]), len(videoA)))
    country.extend([countryA for i in range(len(videoA))])
    videos.extend(list(videoA))

# Pivot Countries
video_country = pandas.DataFrame.from_dict({"video_id": videos, "country": country})[["video_id", "country"]]
video_country = video_country.pivot_table(index="video_id", columns="country", aggfunc=len, fill_value=0.0)
video_country = pandas.DataFrame(video_country.to_records())

video_country.to_csv("video_country_unique.csv", index=False)
