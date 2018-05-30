import re
import pandas
import tensorflow as tf
from polyglot.detect import Detector
from tensorflow.python.platform import gfile
from global_params import Params

def detect_language(mixed_text): 
    multi_lang = Detector(mixed_text, quiet=True).languages
    return [language.name for language in multi_lang if (language.code != "un") & (language.confidence > 10)]

def extract_video(): 
    videoDFs = []
    for each_country in Params.COUNTRY_EN: 
        video_list = pandas.read_csv(Params.DIR_DATA + each_country + "videos.csv", usecols=["video_id", "title", "category_id"])
        video_list["language"] = video_list["title"].apply(detect_language)
        videoDFs.append(video_list)
    # Get only English videos
    video_all = pandas.concat(videoDFs)
    video_all.drop_duplicates(subset="video_id", inplace=True)
    video_english = video_all[video_all.apply(lambda x: x["language"] == ["English"], axis=1)]
    # Cleanse text for language analysis
    video_english.is_copy = False
    title_english = video_english["title"].str.lower()
    title_english = title_english.apply(lambda x: re.sub(r'[^a-zA-Z]', ' ', x))
    title_english = title_english.apply(lambda x: ' '.join([word for word in x.split() if len(word) > 1 or word == "i"]))
    video_english.loc[:,"title"] = title_english

    video_english.drop(columns=["language"]).to_csv(Params.DIR_DATA + Params.FILE_NAME_VIDEO, index=False)
    
def create_vocab(): 
    # Create word index
    video_english = pandas.read_csv(Params.DIR_DATA + Params.FILE_NAME_VIDEO)
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(Params.MAX_DOC_LENGTH)
    vocab_processor.fit(video_english["title"].astype('str'))
    
    with gfile.Open(Params.DIR_DATA + Params.FILE_NAME_VOCAB, 'wb') as f: 
        f.write("{}\n".format(Params.PADWORD))
        for word, index in vocab_processor.vocabulary_._mapping.items():
            f.write("{}\n".format(word))