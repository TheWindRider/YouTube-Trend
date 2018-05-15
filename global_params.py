class Params: 
    # files
    FOLDER_NAME = "data/"
    COUNTRY_ALL = ["CA", "DE", "FR", "GB", "US"]
    COUNTRY_EN = ["CA", "GB", "US"]
    FILE_NAME_CATEGORY = "category_lkp.csv"
    FILE_NAME_COUNTRY = "video_country_unique.csv"
    FILE_NAME_VIDEO = "video_english_unique.csv"
    FILE_NAME_VOCAB = "vocabulary.tsv"
    FILE_NAME_MODEL = "title_category.npz"
    # text data
    MAX_DOC_LENGTH = 15
    PADWORD = 'ZYXW'
    NUM_CLASS = 16
    # neural network
    EMBED_SIZE = 16
    CONV_SIZE = 5
    CONV_STRIDE = 2
    CONV_FILTER = 4
    HIDDEN_FILTER = 64
    # training
    SPLIT_RATE = 0.2
    DROP_RATE = 0.3
    LEARN_RATE = 0.0005
    BATCH_SIZE = 64
    NUM_EPOCH = 50
    NUM_STEP = 1000
