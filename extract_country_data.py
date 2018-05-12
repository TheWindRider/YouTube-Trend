import pandas
from global_params import Params

videoID_list = [pandas.read_csv(Params.FOLDER_NAME + each_country + "videos.csv", usecols=["video_id"]) 
                for each_country in Params.COUNTRY_ALL]
country = []
videos = []

# Extract, Keep Unique
for i in range(len(Params.COUNTRY_ALL)): 
    countryA = Params.COUNTRY_ALL[i]
    videoA = set(videoID_list[i]["video_id"])
    print("{}: {} videos, {} unique".format(countryA, len(videoID_list[i]), len(videoA)))
    country.extend([countryA for i in range(len(videoA))])
    videos.extend(list(videoA))

# Pivot Countries
video_country = pandas.DataFrame.from_dict({"video_id": videos, "country": country})[["video_id", "country"]]
video_country = video_country.pivot_table(index="video_id", columns="country", aggfunc=len, fill_value=0.0)
video_country = pandas.DataFrame(video_country.to_records())

video_country.to_csv(Params.FILE_NAME_COUNTRY, index=False)
