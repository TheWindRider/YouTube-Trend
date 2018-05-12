import pandas
from polyglot.detect import Detector
from collections import Counter
from global_params import Params

def detect_language(mixed_text): 
    multi_lang = Detector(mixed_text, quiet=True).languages
    return [language.name for language in multi_lang if (language.code != "un") & (language.confidence > 10)]

def sample_language(country, language, n_sample=10): 
# useful post-analysis
    video_list = pandas.read_csv(Params.FOLDER_NAME + country + "videos.csv", usecols=["video_id", "title"])
    video_list["language"] = video_list["title"].apply(detect_language)
    video_interest = video_list[video_list.apply(lambda x: language in x["language"], axis=1)]
    return video_interest.sample(n_sample)
    
for each_country in Params.COUNTRY_ALL: 
    video_list = pandas.read_csv(Params.FOLDER_NAME + each_country + "videos.csv", usecols=["video_id", "title"])
    video_list["language"] = video_list["title"].apply(detect_language)
    video_language = []
    for each_language in video_list["language"]: 
        video_language.extend(each_language)
    languages = Counter(video_language)
    print(each_country, languages.most_common(5))
