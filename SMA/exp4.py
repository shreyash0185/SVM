import pickle
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
!pip install emoji
!pip install vaderSentiment
from emoji import UNICODE_EMOJI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google.colab import drive
!pip install TextBlob
from textblob import TextBlob
from textblob import Word
#mount google drive 
drive.mount('/content/drive')
#Download and unzip files
# added example
!ls "/content/drive/My Drive"
!sudo apt-get install unzip
!pwd
####################################
# example text 
file_tina = "/content/drive/My Drive/tinas_comments.csv"
file_ken = "/content/drive/My Drive/kens_comments.csv"
file_debate1 = "/content/drive/My Drive/debate1_comments.csv"

df_tina = pd.read_csv(file_tina)
df_ken = pd.read_csv(file_ken)
df_debate1 = pd.read_csv(file_debate1)
df_tina['channel_name']='Tina Huang'
df_ken['channel_name']='Ken Jee'
df = pd.concat([df_tina, df_ken])
df.head()
comments = df.query('author_name != "Tina Huang" & author_name != "Ken Jee"')
comments.tail()
# choose who's comments 
comments = comments.query('channel_name == "Ken Jee"')
debate_comments = df_debate1
# brief clean up
def preprocess(comment):
    comment = comment.str.replace("\n", " ") # remove new lines 
    return comment

comments['comment'] = preprocess(comments['comment'])
# @Ken Jee https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg
# https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664
sentiment = SentimentIntensityAnalyzer()
sentiment.polarity_scores(comments.comment[0])
comments['vader_sentiment'] = comments.comment.apply(lambda x: sent.polarity_scores(x))
comments['vader_neg_sentiment'] = comments.vader_sentiment.apply(lambda x: x['neg'])
comments['vader_pos_sentiment'] = comments.vader_sentiment.apply(lambda x: x['pos'])
comments['vader_comp_sentiment'] = comments.vader_sentiment.apply(lambda x: x['compound'])
# most positive 
comments.sort_values(by=['vader_comp_sentiment'], ascending=False)[['comment']].head(10)
# most negative 
comments.sort_values(by=['vader_comp_sentiment'], ascending=True)[['comment']].head(10)
## Another method 
## credit: https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
comments['textblob_polarity'] = comments['comment'].map(lambda text: TextBlob(text).sentiment.polarity)
comments['comment_len'] = comments['comment'].astype(str).apply(len)
comments['word_count'] = comments['comment'].apply(lambda x: len(str(x).split()))
# most positive comments
comments.sort_values(by=['textblob_polarity'], ascending=False)[['comment', 'textblob_polarity']].head(10)
# most negative
comments.sort_values(by=['textblob_polarity'], ascending=True)[['comment', 'textblob_polarity']].head(10)
# most neutral 
#questions? 
textblob_neutral = comments.query('textblob_polarity == 0').rename(columns={"comment": "textblob_comment"})[['textblob_comment']].reset_index(drop=True)
vader_neutral = comments.query('vader_comp_sentiment == 0').rename(columns={"comment": "vader_comment"})[['vader_comment']].reset_index(drop=True)
vader_neutral # probably should correct spelling errors with textblob 
# most number of likes
pd.set_option('display.max_colwidth', None)
comments.sort_values(by=['like_count'], ascending=False)[['comment']].head(10)
# comparison of textblob and vader sentiment distribution
sub = comments[['comment', 'textblob_polarity', 'vader_comp_sentiment', 'just_date', 'vidid', 'comment_len', 'word_count']]
pol_hist = sub.melt(id_vars=['comment', 'just_date', 'vidid', 'comment_len', 'word_count'], value_vars=['textblob_polarity', 'vader_comp_sentiment'], var_name='method', value_name='sentiment')
sns.displot(pol_hist, x="sentiment",hue="method", element="step")

# comment_len 
only_textblob = pol_hist.query('method == "textblob_polarity"')
sns.displot(only_textblob, x="comment_len")
sns.displot(only_textblob, x="word_count")
# videos with highest and lowest sentiments 
# highest 
# lowest
comments.groupby('vid_title').mean()[['textblob_polarity']].sort_values(by=['textblob_polarity'], ascending=True)
# sentiment per videos
import matplotlib.pylab as plt
sns.set(rc={'figure.figsize':(11.7,8.27)}) 
ax = sns.boxplot(x="vid_title", y="sentiment", data=vidid_pol_hist)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
# sentiments over time 
sns.set(rc={'figure.figsize':(20,9)})
sns.lineplot(x="just_date", y="sentiment", hue="method", data=pol_hist) 
# https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
!pip install scattertext
import spacy.cli
spacy.cli.download("en_core_web_lg")
import scattertext as st
nlp = spacy.load('en_core_web_lg')
corpus = st.CorpusFromPandas(only_textblob, category_col='vidid', text_col='comment', nlp=nlp).build()
# terms that differentiate text from background text 
print(list(corpus.get_scaled_f_scores_vs_background().index[:20]))
# https://medium.com/@rohithramesh1991/unsupervised-text-clustering-using-natural-language-processing-nlp-1a8bc18b048d
# https://github.com/rohithramesh1991/Text-Preprocessing/blob/master/Text%20Preprocessing_codes.py
# https://stackoverflow.com/questions/51217909/removing-all-emojis-from-text 
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
%matplotlib inline
"""removes punctuation, stopwords, and returns a list of the remaining words, or tokens"""
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer, WordNetLemmatizer

# remove emojis 
def give_emoji_free_text(text):
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def text_process(text):
    # stemmer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
    # temp = [stemmer.lemmatize(word) for word in nopunc]
    temp = [stemmer.stem(word) for word in nopunc]

    joined = ' '.join(temp)
    # remove emojis 
    joined = give_emoji_free_text(joined)

    # correct spelling 
    # joined = str(TextBlob(joined).correct())

    return joined.split()
##kmeans
desc = debate_comments['comment'].values 
vectorizer4 = TfidfVectorizer(analyzer = text_process, stop_words=stopwords.words('english'), ngram_range=(1,3))
X4 = vectorizer4.fit_transform(desc)
words = vectorizer4.get_feature_names()
wcss = []
for i in range(1,10):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X4)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,10),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()
kmeans = KMeans(n_clusters = 2, n_init = 20, n_jobs = 1) # n_init(number of iterations for clsutering) n_jobs(number of cpu cores to use)
kmeans.fit(X4)
# We look at 2 the clusters generated by k-means.
common_words = kmeans.cluster_centers_.argsort()[:,-1:-26:-1]
for num, centroid in enumerate(common_words):
    print(str(num) + ' : ' + ', '.join(words[word] for word in centroid))
     