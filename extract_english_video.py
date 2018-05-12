import re
import pandas
from string import digits
from polyglot.detect import Detector
from global_params import Params

def detect_language(mixed_text): 
    multi_lang = Detector(mixed_text, quiet=True).languages
    return [language.name for language in multi_lang if (language.code != "un") & (language.confidence > 10)]
    
videoDFs = []
for each_country in Params.COUNTRY_EN: 
    video_list = pandas.read_csv(Params.FOLDER_NAME + each_country + "videos.csv", usecols=["video_id", "title", "category_id"])
    video_list["language"] = video_list["title"].apply(detect_language)
    videoDFs.append(video_list)
    
# Get only English videos
video_all = pandas.concat(videoDFs)
video_all.drop_duplicates(subset="video_id", inplace=True)
video_english = video_all[video_all.apply(lambda x: x["language"] == ["English"], axis=1)]

# Cleanse text for language analysis
video_english.is_copy = False
title_english = video_english["title"].str.lower()
title_english = title_english.apply(lambda x: x.translate(str.maketrans('', '', digits)))
title_english = title_english.apply(lambda x: re.sub(r'[^\w]', ' ', x))
title_english = title_english.apply(lambda x: ' '.join(x.split()))
video_english.loc[:,"title"] = title_english

video_english.drop(columns=["language"]).to_csv(Params.FILE_NAME_VIDEO, index=False)
