
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
import tweepy
import requests
import time
import matplotlib.pyplot as plt
import re

pd.set_option('max_colwidth',-1)


# ## Gather ##

# In[2]:


#读取csv 文件
archive = pd.read_csv('twitter-archive-enhanced.csv')


# In[3]:


#读取 tsv 
image = pd.read_csv('image-predictions.tsv',sep='\t')


# In[3]:


#从网页下载数据
url = 'https://raw.githubusercontent.com/udacity/new-dand-advanced-china/master/%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97/WeRateDogs%E9%A1%B9%E7%9B%AE/image-predictions.tsv'
response = requests.get(url)
#保存到文件夹
with open('image-predictions.tsv',mode='wb') as file:
    file.write(response.content)


# In[5]:


consumer_key = 'YOUR CONSUMER KEY'
consumer_secret = 'YOUR CONSUMER SECRET'
access_token = 'YOUR ACCESS TOKEN'
access_secret = 'YOUR ACCESS SECRET'


# In[6]:


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# In[ ]:


df_list = []

start = time.time()

for i in archive['tweet_id']:
    try:
        tweet = api.get_status(i, tweet_mode='extended')
        df_list.append(tweet._json)
        
    except tweepy.TweepError as e:
        print (i, ":", e)
        
end = time.time()
print(end - start)


# In[ ]:


with open('tweet_json.txt','w') as out_file:
    for i in df_list:
        json.dump(i, out_file)
        out_file.write('\n')


# In[4]:


#读取txt 文件
tweet = pd.read_json('tweet_json.txt',lines=True,encoding = 'utf-8')
tweet.columns


# In[5]:


#选择需要的列
tweets = tweet[['id','retweet_count','favorite_count']]
tweets.head()


# In[6]:


archive.head()


# In[7]:


archive.tail()


# In[7]:


archive.info()


# In[8]:


archive.describe()


# In[9]:


archive[archive.text.str.contains('&amp;')]


# In[10]:


archive.name.value_counts()


# In[11]:


archive.source.value_counts()


# In[12]:


archive.name.unique()


# In[13]:


#找到name列为小写的数据
archive.loc[(archive['name'].str.islower())]


# In[14]:


len(archive.loc[(archive['name'].str.islower())])


# In[15]:


#text中包含狗的名字的数据 named
archive.loc[(archive['name'].str.islower())&(archive['text'].str.contains('named'))]


# In[16]:


len(archive.loc[(archive['name'].str.islower())&(archive['text'].str.contains('named'))])


# In[17]:


#text中包含狗的名字的数据 name is
archive.loc[(archive['name'].str.islower())&(archive['text'].str.contains('name is'))]


# In[18]:


len(archive.loc[(archive['name'].str.islower())&(archive['text'].str.contains('name is'))])


# In[22]:


image.head()


# In[23]:


image.tail()


# In[24]:


image.info()


# In[25]:


tweets.tail()


# In[26]:


tweets.info()


# In[27]:


tweets.describe()


# # Quality #
# ## archive ##
# ・错误的数据类型 : 'tweet_id''timestamp' 
# 
# ・Rating （评分） 包含不准确数据，查找更新，合并分子分母用rating来表示它们的比例
# 
# ・包含了retweets ，在分析中不需要 : 
#       先将 retweeted_status_id 不为空的所有行删除，若这一列数据全部为空，删除掉整列
#       'retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'，'in_reply_to_status_id','in_reply_to_user_id' 
# 
# ・Source列包含不需要的内容  
# 
# ・Name列包含不正确的数据 (eg. 'a','the','an'，None...)  
# 
# ・需要删除无图片数据
# ## tweets ##
# ・错误的数据类型 :tweet_id 
# 
# ## image ##
# ・错误的数据类型 :id  
# 
# ・p1 & p2 & p3的数据首字母大小写不统一，p2和p3的内容在分析中似乎不必要
# 
# 
# # Tidiness #
# ## archive ##
# ・狗的地位分为4列，要整合在一列 
# 
# ・把tweets和image合并到archive   

# 
# # Clean #

# In[8]:


archive_clean = archive.copy()
tweets_clean = tweets.copy()
image_clean = image.copy()


# In[9]:


archive_clean.info()


# ### Define ###
# 
# ・包含了retweets ，在分析中不需要，应删除 :
# ['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp']
# 

# ### Code ###

# In[10]:


archive_clean = archive_clean[archive_clean.retweeted_status_id.isnull()]
archive_clean = archive_clean[archive_clean.in_reply_to_user_id.isnull()]


# In[11]:


archive_clean = archive_clean.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'],axis=1)


# ### Test ###

# In[12]:


archive_clean.info()


# ### Define ###
# 
# 狗的地位分为4列，要整合在一列

# ### Code ###

# In[21]:


def get_stages(row):
    stages = []
    all_stages = ['doggo','floofer','pupper','puppo']
    for i in all_stages:
        if row[i] != 'None':
            stages.append(i)
            
    return ','.join(sorted(stages))

archive_clean['dog_stage'] = archive_clean[['doggo','floofer','pupper','puppo']].apply(get_stages, axis=1)


# In[22]:


archive_clean =  archive_clean.drop(['doggo','floofer','pupper','puppo'], axis=1)


# In[23]:


archive_clean['dog_stage'] = archive_clean['dog_stage'].astype('category')


# ### Test ###

# In[24]:


archive_clean.dog_stage.value_counts()


# In[25]:


archive_clean.sample(20)


# ### Define ###
# 把tweets和image合并到archive

# ### Code ###

# In[26]:


tweets_clean =  tweets_clean.rename(columns={'id':'tweet_id','retweet_count':'retweet_count','favorite_count':'favorite_count'})


# In[27]:


archive_clean = archive_clean.merge(tweets_clean, on='tweet_id', how='inner')


# In[28]:


archive_clean = archive_clean.merge(image_clean, on='tweet_id', how='inner')


# ### Test ###

# In[29]:


archive_clean.sample(20)


# In[30]:


archive_clean.info()


# ### Define ###
# 
# Name列包含不正确的数据 (eg. 'a','the','an'，None...)

# ### Code ###

# In[31]:


#找到name小写的行
archive_clean['name'].str.islower().value_counts()


# In[32]:


#text中存在 named 或者 name is的数据
named = archive_clean.loc[(archive_clean['name'].str.islower())&(archive_clean['text'].str.contains('named'))]
name_is = archive_clean.loc[(archive_clean['name'].str.islower())&(archive_clean['text'].str.contains('name is'))]
no_name = archive_clean.loc[(archive_clean['name'].str.islower())]


# In[33]:


#把text存到list
named_l = named['text'].tolist()
name_is_l = name_is['text'].tolist()
no_name_l = no_name['text'].tolist()


# In[34]:


#提取text中的name，更新nema列
for i in named_l:
    w = archive_clean.text == i
    name_column = 'name'
    archive_clean.loc[w,name_column] = re.findall(r"named\s(\w+)", i)


# In[35]:


for i in name_is_l:
    w = archive_clean.text == i
    name_column = 'name'
    archive_clean.loc[w,name_column] = re.findall(r"name is\s(\w+)",i)


# In[36]:


for i in no_name_l:
    w = archive_clean.text ==i
    name_column = 'name'
    archive_clean.loc[w,name_column] = "None"


# ### Test ###

# In[37]:


archive_clean['name'].str.islower().value_counts()


# In[38]:


archive_clean.name.value_counts()


# ### Define ###
# Source列包含不需要的内容，刪除简化

# ### Code ###

# In[39]:


archive_clean.source = archive_clean.source.replace('<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>','Twitter for iPhone')
archive_clean.source = archive_clean.source.replace('<a href="http://twitter.com" rel="nofollow">Twitter Web Client</a>','Twitter Web Client')
archive_clean.source = archive_clean.source.replace('<a href="https://about.twitter.com/products/tweetdeck" rel="nofollow">TweetDeck</a>','TweetDeck')


# ### Test ###

# In[40]:


archive_clean.source.value_counts()


# ### Define ###
# p1 & p2 & p3的数据首字母大小写不统一，p2和p3的内容在分析中似乎不必要

# ### Code ###

# In[41]:


#首字母大写
archive_clean.p1 = archive_clean.p1.str.capitalize()
archive_clean.p2 = archive_clean.p2.str.capitalize()
archive_clean.p3 = archive_clean.p3.str.capitalize()


# ### Test ###

# In[42]:


archive_clean.p1.str.islower().value_counts()


# In[43]:


archive_clean.p2.str.islower().value_counts()


# In[44]:


archive_clean.p3.str.islower().value_counts()


# ### Define ###
# Rating （评分） 包含不准确数据，查找更新，合并分子分母用rating来表示它们的比例
# 

# ### Code ###

# In[45]:


archive_clean.rating_numerator.value_counts()


# In[46]:


archive_clean.rating_denominator.value_counts()


# In[47]:


#匹配找出text中出现两个及以上分母的行
fix_rating = archive_clean[archive_clean.text.str.contains(r"(\d+\.?\d*\/\d+\.?\d*\D+\d+\.?\d*\/\d+\.?\d*)")]


# In[48]:


fix_rating


# In[49]:


#保存到list
fix_rating_l = ['After so many requests, this is Bretagne. She was the last surviving 9/11 search dog, and our second ever 14/10. RIP https://t.co/XAVDNDaVgQ',
'Happy 4/20 from the squad! 13/10 for all https://t.co/eV1diwds8a',
'This is Bluebert. He just saw that both #FinalFur match ups are split 50/50. Amazed af. 11/10 https://t.co/Kky1DPG4iq',
'This is Darrel. He just robbed a 7/11 and is in a high speed police chase. Was just spotted by the helicopter 10/10 https://t.co/7EsP8LmSp5',
'This is an Albanian 3 1/2 legged Episcopalian. Loves well-polished hardwood flooring. Penis on the collar. 9/10 https://t.co/d9NcXFKwLv']


# In[50]:


#提取text中的第二个分数的分子，然后更新分子，分母改为10 
for i in fix_rating_l:
    w = archive_clean.text == i
    cnum = 'rating_numerator'
    cdeno = 'rating_denominator'
    archive_clean.loc[w,cnum] = re.findall(r"\d+\.?\d*\/\d+\.?\d*\D+(\d+\.?\d*)\/\d+\.?\d*", i)
    archive_clean.loc[w,cdeno] = 10


# ### Test ###
# 

# In[51]:


archive_clean[archive_clean.text.isin(fix_rating_l)]


# ### Code ###

# In[52]:


#匹配存在小数的评分
fix_rating1 = archive_clean[archive_clean.text.str.contains(r"(\d+\.\d*\/\d+)")]
fix_rating1


# In[53]:


fix_text_l = fix_rating1['text'].tolist()


# In[54]:


for i in fix_text_l:
    w = archive_clean.text == i
    cnum = 'rating_numerator'
    cdeno = 'rating_denominator'
    archive_clean.loc[w,cnum] = re.findall(r"(\d+\.\d*)", i)
    archive_clean.loc[w,cdeno] = 10


# In[55]:


#更新数据类型
archive_clean['rating_numerator'] = archive_clean['rating_numerator'].astype('float')
archive_clean['rating_denominator'] = archive_clean['rating_denominator'].astype('float')


# In[56]:


#合并分子分母用rating来表示它们的比例
archive_clean['ratings'] = archive_clean.rating_numerator/archive_clean.rating_denominator


# ### Test ###

# In[57]:


archive_clean[archive_clean.text.isin(fix_text_l)]


# In[58]:


archive_clean.head()


# In[59]:


archive_clean.info()


# ### Define ###
# 存在错误的数据类型，更改数据类型

# ### Code ###

# In[60]:


archive_clean['tweet_id'] = archive_clean['tweet_id'].astype('str')
archive_clean.timestamp = pd.to_datetime(archive_clean.timestamp)
archive_clean['source'] = archive_clean['source'].astype('category')
archive_clean['text'] = archive_clean['text'].astype('str')


# ### Test ###

# In[61]:


archive_clean.info()


# In[62]:


archive_clean.head()


# ## Store ##
# 

# In[63]:


archive_clean.to_csv('twitter_archive_master.csv')


# ## Analyzing ##
# 

# In[64]:


import datetime
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[65]:


#读取文件
df =pd.read_csv('twitter_archive_master.csv').drop(['Unnamed: 0'],axis=1)


# In[66]:


df.columns


# In[87]:


anadf = df[['tweet_id', 'timestamp', 'retweet_count', 'favorite_count', 'ratings', 'p1', 'p1_conf']].copy()


# In[88]:


len(anadf.p1.value_counts())


# 通过统计推文p1的预测来得出该统计中狗的品种为378种，狗的品种真的好多啊，那么我们将通过得出数量前十位的通过转发量，点赞数和评分来得到最有人气的狗狗品种。

# In[96]:


top10_vc = anadf.p1.value_counts().head(10).index
top10_vc


# In[97]:


top10_vcdog =  anadf.loc[anadf.p1.isin(top10_vc)]
top10_vcdog.head()


# In[98]:


top10_vcdog.p1.value_counts()


# In[99]:


mean = top10_vcdog.groupby(top10_vcdog['p1']).mean().sort_values('favorite_count')


# In[100]:


mean


# 对每个品种的狗的转发量，点赞数，评分求平均数，然后进行排序。

# In[101]:


x = mean.index
y1 = mean['favorite_count']

plt.barh(x, y1, data=mean)
plt.xlabel('Mean of favorite count')
plt.ylabel('Dog breeds')
plt.title('The Mean of favorite count top10 dog-breeds');


# 这幅图可以看出获得平均点赞数最高的品种是萨摩耶。

# In[102]:


x = mean.index
y1 = mean['retweet_count']

plt.barh(x, y1, data=mean)
plt.xlabel('Mean of retweet count')
plt.ylabel('Dog breeds')
plt.title('The Mean of retweet count top10 dog-breeds');


# 同样可以看出萨摩耶的平均转发数是最高的。

# In[103]:


x = mean.index
y1 = mean['ratings']
plt.barh(x, y1, data=mean)
plt.xlabel('Mean of ratings')
plt.ylabel('Dog breeds')
plt.title('The Mean of ratings top10 dog-breeds');


# 平均评级中最高的还是萨摩耶哟。

# 根据上面的统计数据和图表，我们可以看到萨摩耶拥有最多的平均转推次数和最多点赞数。所以可以说萨摩耶是最有人气的品种。
# 
# 萨摩耶憨憨样子真的很可爱，但是它是一直都有这么高的人气吗？那么接下来我们来看看大家对萨摩耶的好感度呈现什么样的变化趋势吧。
# 
# 首先提取出来推文中关于萨摩耶的所有数据，然后绘制散点图。

# In[104]:


samoyed = anadf[anadf['p1'] == 'Samoyed']


# 我认为并不需要精确到分秒的时间，所以只把日期提取出来就好了。

# In[105]:


samoyed.timestamp = pd.to_datetime(samoyed.timestamp)


# In[106]:


samoyed['datetime'] = samoyed['timestamp'].map(lambda x: x.strftime('%Y-%m-%d'))
samoyed = samoyed.drop(['timestamp'],axis=1)


# In[107]:


samoyed = samoyed.sort_values(by='datetime',ascending=True)


# In[108]:


x = samoyed['datetime']
y = samoyed['ratings']

plt.scatter(x, y, s=20,c='g')
plt.xlabel('datetime')
plt.ylabel('rating')
plt.ylim([0.6,1.5])
plt.xticks(np.arange(2,44,10))
plt.title('The trend of ratings about Samoyed');


# 这是对萨摩耶的rating的变化图，rating是在前面我们计算出的分子分母的比例，整体变化不是很大平均都大于1，也就是说大家对它的评分大多超过10分呢。

# In[109]:


x = samoyed['datetime']
y1 = samoyed['retweet_count']
y2 = samoyed['favorite_count']

plt.scatter(x, y1, s=20,c='b')
plt.scatter(x, y2, s=20,c='r')
plt.xlabel('datetime')
plt.ylabel('counts')
plt.legend(loc='upper left')
plt.xticks(np.arange(2,44,10))
plt.title('The trend of retweet and favorite about Samoyed');


#   这个是大家对萨摩耶转发量和点赞数量的变化趋势图，这个图可以更容易的看出从2015年开始到2017年之间，变化是明显增加的趋势，尤其点赞量的增长更加显著。
# 萨摩耶狗狗看起来好像永远在笑的样子，能给大家带来好心情，面对快节奏生活的压力的人们看到萨摩耶可爱的笑容会被瞬间治愈，它们又非常喜欢亲近人类，我认为这可能是萨摩耶越来越受大家喜爱的原因吧！
