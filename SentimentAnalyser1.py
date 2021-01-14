import pandas as pd
import pickle
import gensim
import keras
import numpy as np
import tweepy
import xgboost as xgb
from keras.preprocessing.sequence import pad_sequences
from preprocessing import vectorize_tweet
from searchtweets import gen_rule_payload, load_credentials, collect_results, ResultStream
import nltk
from visualiser import Visualiser
from StreamListener import StreamListener
import time
from datetime import datetime
import glob

class Analyser:

    def __init__(self):
        '''
        :param model_type: model used, at the moment lstm and xgb are supported
        '''
        if not nltk.download('punkt') : nltk.download('punkt')
        self._auth = False
        self._data = pd.DataFrame()
        self._scored = False


    def twitter_login(self, ACCESS_TOKEN, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET):
        '''
        :param ACCESS_TOKEN:
        :param ACCESS_TOKEN_SECRET:
        :param CONSUMER_KEY:
        :param CONSUMER_SECRET:
        :return:
        '''
        # Create login for search by users and search by words
        self._auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        self._auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        # Create credential variable for the historical API search
        self._premium_search_args = load_credentials(r"data/login.yaml",
                                               yaml_key="search_tweets_premium", env_overwrite=False)

    def get_tweets(self,
                   inclusions=None,
                   by='all',
                   exclusions=None,
                   connector = "OR",
                   time_limit=40,
                   item_limit=10,
                   exclude_retweets=False,
                   lang=['ar', 'en'],
                   from_date=None,
                   to_date=None,
                   country = np.nan,
                   keyword = np.nan,
                   archive =  False):
        '''
        :param inclusions: Words or Users to be scanned depending on the parameters[array]
        :param Words: If True, filter by words
        :param Users: If True, stream by users
        :param item_limit: limit the number of tweets when filtering by users
        :return: returns pd dataframe
        '''
        self._scored = False

        if (lang != 'ar') & (lang !='en'):
            raise Exception("Unsuported language in list; we presently only support 'en' and 'ar'")

        # if len(set(lang).difference({'en', 'ar'})) > 0:
        #     raise Exception("Unsuported language in list; we presently only support 'en' and 'ar'")

        if (connector != 'AND') & (connector !='OR'):
            raise Exception("Unsuported connector in list; we presently only support 'OR' and 'AND'")

        if self._auth is False:
            raise Exception('Use twitter_login method first')

        if inclusions is None and by is not None:
            raise Exception('Please specify the desired inclusions in a list')

        API = tweepy.API(self._auth, wait_on_rate_limit=True, retry_errors={401, 404, 500, 503})
        listener = StreamListener(api=API, time_limit=time_limit, exclusions=exclusions, lang=lang, archive=archive)
        streamer = tweepy.Stream(auth=self._auth, listener=listener)

        if by == "stream_words":
            print("Streaming tweets by keywords")
            streamer.filter(track=inclusions)
            self._data = listener._tweets.reset_index(drop=True)

            if exclude_retweets:
                self._data = self._data[self._data['retweeted'] == False]


        elif by == "users":

            print("Filtering by users")
            df = pd.DataFrame(
                columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location", "retweeted", "country", "keyword"])
            skipped = 0
            skipped_u = 0
            for user in inclusions:
                try :
                    for twt in tweepy.Cursor(API.user_timeline, id=user, count = 200, tweet_mode = 'extended').items(item_limit):
                        if ((not twt.retweeted) and ('RT @' not in twt.full_text)) or not exclude_retweets:
                            try:
                                pg_df = pd.json_normalize(twt._json).reset_index(drop=True).drop_duplicates(subset='id')
                                pg_df['screen_name'] = twt.user.screen_name
                                pg_df['user_lang'] = twt.user.lang
                                pg_df['tweet'] = twt.full_text
                                pg_df['location'] = twt._json['user']['location']
                                pg_df['country'] = country
                                pg_df['keyword'] = keyword
                                pg_df = pg_df[["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location", "retweeted", "country", "keyword"]]
                                df = pd.concat([df, pg_df], axis=0).dropna(subset=['id']).drop_duplicates(subset='id')
                            except:
                                skipped+=1

                    self._data = df.reset_index(drop=True).drop_duplicates(subset='id')
                except:
                    print('Skipped a user due to an exception, sleeping 60 secs')
                    time.sleep(60)
                if archive : pickle.dump(self._data, open("data/archive/"+by+"/"+user+"_"+datetime.now().strftime("%Y%m%d")+".pkl", 'wb'))


            print('{} invalid tweets skipped'.format(skipped))
            print('{} invalid users skipped'.format(skipped_u))

        elif by == "countries":


            print("Filtering by countries")
            # Generate Rule for searching. Inclusion is query of the request and from_date/to_date must be specified
            q = "place_country:QA"
            rule = gen_rule_payload(q, results_per_call=1)
            tweets = collect_results(rule, max_results=item_limit, result_stream_args=self._premium_search_args)
            df = pd.DataFrame(
                columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location","hashtags", "mentions", "retweeted"])
            # Converting the result into a data frame that is consistent with other methods.

            result_list = [[t.all_text.replace('\n', '||').replace('\r\n', '||'),
                            t.id,
                            t.created_at_datetime,
                            t.lang,
                            t['user']['lang'],
                            t.screen_name,
                            t['user']['location'],
                            [tag['text'] for tag in t['entities']['hashtags']],
                            [tag['screen_name'] for tag in t['entities']['user_mentions']],
                            t['retweeted']] for t in tweets]

            
            df = pd.DataFrame(result_list,columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location","hashtags", "mentions", "retweeted"])
            self._data = df.reset_index(drop=True).dropna(subset=['id']).drop_duplicates(subset='id')
            if archive: pickle.dump(self._data, open(
                "data/archive/countries/" + datetime.now().strftime("%Y%m%d_%I%M%p") + ".pkl", 'wb'))


            print('{} invalid tweets skipped'.format(skipped))

        elif by == "keywords":

            print("Filtering by keywords")
            # Generate Rule for searching. Inclusion is query of the request and from_date/to_date must be specified
            q = "("+connector.join(inclusions)+") lang:"+lang
            rule = gen_rule_payload(q, results_per_call=100)
            tweets = collect_results(rule, max_results=item_limit, result_stream_args=self._premium_search_args)
            df = pd.DataFrame(
                columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location","hashtags", "mentions", "retweeted"])
            # Converting the result into a data frame that is consistent with other methods.

            result_list = [[t.all_text.replace('\n', '||').replace('\r\n', '||'),
                            t.id,
                            t.created_at_datetime,
                            t.lang,
                            t['user']['lang'],
                            t.screen_name,
                            t['user']['location'],
                            [tag['text'] for tag in t['entities']['hashtags']],
                            [tag['screen_name'] for tag in t['entities']['user_mentions']],
                            t['retweeted']] for t in tweets]

            df = pd.DataFrame(result_list,columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location","hashtags", "mentions", "retweeted"])
            self._data = df.reset_index(drop=True).dropna(subset=['id']).drop_duplicates(subset='id')
            if archive: pickle.dump(self._data, open(
                "data/archive/keywords/" + datetime.now().strftime("%Y%m%d_%I%M%p") + ".pkl", 'wb'))


        elif by == 'hist_words':

            print("Filtering by historical keywords")

            # Generate Rule for searching. Inclusion is query of the request and from_date/to_date must be specified
            rule = gen_rule_payload(inclusions, from_date=from_date, to_date=to_date, results_per_call=10)
            tweets = collect_results(rule, max_results=item_limit, result_stream_args=self._premium_search_args )

            # Converting the result into a data frame that is consistent with other methods.
            result_list = [[t.all_text.replace('\n', '||').replace('\r\n', '||'),
                            t.id,
                            t.created_at_datetime,
                            t.lang,
                            t['user']['lang'],
                            t.screen_name,
                            t['user']['location'],
                            [tag['text'] for tag in t['entities']['hashtags']],
                            t['retweeted']] for t in tweets]

            df = pd.DataFrame(result_list, columns=["tweet", "id", "created_at", "lang", "user_lang", "screen_name", "location", "hashtags", "retweeted"])
            self._data = df.reset_index(drop=True).dropna(subset=['id']).drop_duplicates(subset='id')
            if archive: pickle.dump(self._data, open(
                "data/archive/keywords/" + datetime.now().strftime("%Y%m%d_%I%M%p") + ".pkl", 'wb'))

        elif by == 'all':
            print("No Filter")
            streamer.sample()
            self._data = listener._tweets.reset_index(drop=True)
        else:
            raise Exception("Please assign valid value to by parameter ['all','words','users','stream_words']")

    def set_tweets(self, tweets, lang = 'en'):
        self._data = pd.DataFrame()
        self._data['tweet'] = tweets
        self._data['lang'] = lang

    def tweets(self):
        return self._data

    def score_tweets(self, model_type = None):
        
        if model_type == 'lstm':
            self.models_ar = pickle.load(open('data/ar_models_LSTM.pkl', 'rb'))
            self.models_en = pickle.load(open('data/en_models_LSTM.pkl', 'rb'))

        elif model_type == 'xgb':
            self.models_ar = pickle.load(open('data/ar_models.pkl', 'rb'))
            self.models_en = pickle.load(open('data/en_models.pkl', 'rb'))

        else:
            print('Invalid type specified, using default: XGB')
            self.models_ar = pickle.load(open('data/ar_models.pkl', 'rb'))
            self.models_en = pickle.load(open('data/en_models.pkl', 'rb'))
        if len(self._data) == 0:
            raise Exception('Use get_tweets method first')

        print('Vectorizing Tweets...')

        word_model_ar = None
        word_model_en = None
        if len(self._data.loc[self._data['lang'] == 'ar']) > 0:
            word_model_ar = gensim.models.Word2Vec.load("data/WordVectors/Twt-CBOW")
        if len(self._data.loc[self._data['lang'] != 'ar']) > 0:
            word_model_en = gensim.models.KeyedVectors.load_word2vec_format(
            "data/WordVectors/GoogleNews-vectors-negative300.bin",
            binary=True,
            limit=500000)

        self._data["vectorized_tweet"] = self._data[['tweet','lang']].apply(
            lambda x: vectorize_tweet(
                tweet=x['tweet'].replace('\n', '||').replace('\r\n', '||'),
                modelen=word_model_en,
                modelar=word_model_ar,
                lang=x['lang']),
                axis=1)

        print('Removing invalid entries...')
        nb = len(self._data)
        self._data = self._data.loc[self._data['vectorized_tweet'].apply(lambda x: len(x)) > 0]
        print('{} invalid entries found'.format(nb - len(self._data)))
        if len(self._data) == 0:
            raise Exception('No valid tweets remaining')

        print('Scoring Model...')
        if len(self._data.loc[self._data['lang'] == 'ar']) > 0 :
            for model in self.models_ar:
                if isinstance(self.models_ar[model], xgb.core.Booster):
                    preds = self.models_ar[model].predict(
                    xgb.DMatrix(
                    pd.DataFrame.from_records(
                        self._data.loc[self._data['lang'] == 'ar']["vectorized_tweet"].apply(
                            lambda x: np.mean(x, axis=0)[0]).reset_index(drop = True))))
                    self._data.loc[self._data['lang'] == 'ar', model] = preds



                elif isinstance(self.models_ar[model], keras.engine.sequential.Sequential):

                    preds  = self.models_ar[model].predict(
                        pad_sequences(
                            self._data.loc[self._data['lang'] == 'ar']["vectorized_tweet"].apply(
                                lambda x: np.mean(x, axis=1)),
                            maxlen=40,
                            dtype='float'))
                    self._data.loc[self._data['lang'] == 'ar', model] = preds

        if len(self._data.loc[self._data['lang'] != 'ar']) > 0:
            for model in self.models_en:
                if isinstance(self.models_en[model], xgb.core.Booster):
                    preds = self.models_en[model].predict(
                    xgb.DMatrix(
                    pd.DataFrame.from_records(
                        self._data.loc[self._data['lang'] != 'ar']["vectorized_tweet"].apply(
                            lambda x: np.mean(x, axis=0)[0]).reset_index(drop = True))))
                    self._data.loc[self._data['lang'] != 'ar', model] = preds

                elif isinstance(self.models_en[model], keras.engine.sequential.Sequential):

                    preds  = self.models_en[model].predict(
                        pad_sequences(
                            self._data.loc[self._data['lang'] != 'ar']["vectorized_tweet"].apply(
                                lambda x: np.mean(x, axis=1)),
                            maxlen=40,
                            dtype='float'))
                    self._data.loc[self._data['lang'] != 'ar', model] = preds


        self._data.drop('vectorized_tweet', axis=1, inplace=True)
        self._scored = True
        return self._data

    def overview_chart(self):
        if not self._scored : self.score_tweets()
        sentiments = [sentiment for sentiment in self.models_ar]
        Visualiser(self._data).overview(sentiments=sentiments)

    def pie_chart(self,
                  treshold_pos=0.7,
                  treshold_neg=0.3,
                  positive_sentiments=['joy', 'love'],
                  negative_sentiments=['anger', 'sadness']):
        '''

        :param treshold_pos:
        :param treshold_neg:
        :param positive_sentiments:
        :param negative_sentiments:
        :return:
        '''

        if not self._scored : self.score_tweets()

        Visualiser(self._data).pie_chart(
            treshold_pos=treshold_pos,
            treshold_neg=treshold_neg,
            positive_sentiments=positive_sentiments,
            negative_sentiments=negative_sentiments)

    def bubble_chart(self,
                     rounding=2,
                     positive_sentiments=['joy', 'love'],
                     negative_sentiments=['anger', 'sadness']):
        '''

        :param rounding:
        :param positive_sentiments:
        :param negative_sentiments:
        :return:
        '''

        if not self._scored : self.score_tweets()

        Visualiser(self._data).bubble_chart(
            rounding=rounding,
            positive_sentiments=positive_sentiments,
            negative_sentiments=negative_sentiments)

    def print_top(self, sentiments=[], n=10):
        '''

        :param sentiments:
        :param n:
        :return:
        '''

        if not self._scored : self.score_tweets()

        if len(set(sentiments).difference(self._data.columns)) != 0 or not isinstance(sentiments, list) or len(
                sentiments) == 0:
            raise Exception("Use a valid sentiment list {}".format([sentiment for sentiment in self.models_ar]))
        Visualiser(self._data).print_top(sentiments=sentiments, n=n)

    def collect_archive(self, archive, dedupe = True, score = False):

        path = 'data/archive/'+archive+'/'
        filenames = glob.glob(path + "*.pkl")

        dfs = []
        for filename in filenames:
            dfs.append(pickle.load(open(filename, 'rb')))
        if dedupe : self._data =  pd.concat(dfs, ignore_index=True).dropna(subset=['tweet']).drop_duplicates('id').reset_index(drop=True)
        else : self._data =  pd.concat(dfs, ignore_index=True).dropna(subset=['tweet']).reset_index(drop=True)
        if score :
            score_archive(archive, model_type='xgb')
        self.models_ar = pickle.load(open('data/ar_models.pkl', 'rb'))
        self.models_en = pickle.load(open('data/en_models.pkl', 'rb'))
        try:
            self._data['joy']
            self._scored = True
        except:
            pass

def collect_archive(archive, dedupe = True):

    path = 'data/archive/'+archive+'/'
    filenames = glob.glob(path + "*.pkl")

    dfs = []
    for filename in filenames:
        dfs.append(pickle.load(open(filename, 'rb')))
    if dedupe : data =  pd.concat(dfs, ignore_index=True).dropna(subset=['tweet']).drop_duplicates('id').reset_index(drop=True)
    else : data =  pd.concat(dfs, ignore_index=True).dropna(subset=['tweet']).reset_index(drop=True)
    return data

def score_archive(archive, model_type = 'xgb'):
    path = 'data/archive/' + archive + '/'
    filenames = glob.glob(path + "*.pkl")

    A = Analyser()

    def apply_int(string):
        try:
            return int(string)
        except:
            return np.nan

    for filename in filenames:
        print("Scoring {}".format(filename))
        A._data = pickle.load(open(filename, 'rb'))
        if 'joy' in A._data.columns:
            print("Already scored")
        else :
            A.score_tweets(model_type)
            A._data['id'] = A._data['id'].apply(apply_int)
            pickle.dump(A._data.dropna(subset=['id']).drop_duplicates(subset='id'), open(filename, 'wb'))

