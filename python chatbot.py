#!/usr/bin/env python
# coding: utf-8

# In[71]:


#-*-coding:utf-8-*-
import pandas as pd

DATA_IN_PATH = '/Users/hyeongdoson/Downloads/'

data = pd.read_csv(DATA_IN_PATH + 'ChatbotData_final.csv', encoding ='utf-8')
print(data.head())

sentences = list(data['Q'] + list(data['A']))


# In[51]:


tokenized_sentences = [s.split() for s in sentences]
sent_len_by_token = [len(t) for t in tokenized_sentences]
sent_len_by_eumjeo1 = [len(s.replace(' ', '')) for s in sentences]

from konlpy.tag import Twitter
from konlpy.tag import Okt
okt = Okt()

morph_tokenized_sentences = [okt.morphs(s.replace(' ', '')) for s in sentences]
sent_len_by_morph = [len(t) for t in morph_tokenized_sentences]


# In[52]:


import matplotlib.pyplot as plt

plt.figure(figsize = (12, 5))
plt.hist(sent_len_by_token, bins = 50, range = [0, 50], alpha = 0.5, color = 'r', label = 'eojeol')
plt.hist(sent_len_by_morph, bins = 50, range=[0,50], alpha = 0.5, color='g', label ='morph')
plt.hist(sent_len_by_eumjeo1, bins=50, range = [0, 50], alpha = 0.5, color = 'b', label = 'eumjeol')
plt.title('Sentence Lenth Histogram')
plt.xlabel('Sentence Lenth')
plt.ylabel('Number of Sentences')


# In[53]:


plt.figure(figsize = (12, 5))
plt.hist(sent_len_by_token, bins = 50, range = [0, 50], alpha = 0.5, color = 'r', label = 'eojeol')
plt.hist(sent_len_by_morph, bins = 50, range=[0,50], alpha = 0.5, color='g', label ='morph')
plt.hist(sent_len_by_eumjeo1, bins=50, range = [0, 50], alpha = 0.5, color = 'b', label = 'eumjeol')
plt.title('Sentence Lenth Histogram')
plt.yscale('log')
plt.title('Sentence Lenth Histogram by Eojeol Token')
plt.xlabel('Sentence Lenth')
plt.ylabel('Number of Sentences')


# In[54]:


import numpy as np

print('어절 최대 길이 : {}'.format(np.max(sent_len_by_token)))
print('어절 최소 길이 : {}'.format(np.min(sent_len_by_token)))
print('어절 평균 길이 : {:.2f}'.format(np.mean(sent_len_by_token)))
print('어절 길이 표준편차 : {:.2f}'.format(np.std(sent_len_by_token)))
print('어절 중간 길이 : {}'.format(np.median(sent_len_by_token)))
print('제1사분위 길이 : {}'.format(np.percentile(sent_len_by_token, 25)))
print('제3사분위 길이 : {}'.format(np.percentile(sent_len_by_token, 75)))


# In[55]:


plt.figure(figsize = (12,5))
plt.boxplot([sent_len_by_token, sent_len_by_morph, sent_len_by_eumjeo1],
           labels  = ['Eojeol', 'Morph', 'Eumjeol'],
           showmeans = True)


# In[ ]:


query_sentences = list(data['Q'])
answer_sentences = list(data['A'])

query_morph_tokenized_sentences = [okt.morphs(s.replace(' ', ''))
                                   for s in query_sentences]
query_sent_len_by_morph = [len(t) for t in query_morph_tokenized_sentences]

answer_morph_tokenized_sentences = [okt.morphs(s.replace(' ',''))
                                     for s in answer_sentences]
answer_sent_len_by_morph = [len(t) for t in answer_morph_tokenized_sentences]


# In[ ]:


plt.figure(figsize = (12, 5))
plt.hist(query_sent_len_by_morph, bins = 50, range = [0, 50], alpha = 0.5, color = 'g', label = 'Query')
plt.hist(answer_sent_len_by_morph, bins = 50, range=[0,50], alpha = 0.5, color='r', label ='Answer')
plt.legend()
plt.title('Query Lenth Histogram by Morph Token')
plt.xlabel('Query Lenth')
plt.ylabel('Number of Queries')


# In[ ]:


plt.figure(figsize = (12, 5))
plt.hist(query_sent_len_by_morph, bins = 50, range = [0, 50], alpha = 0.5, color = 'g', label = 'Query')
plt.hist(answer_sent_len_by_morph, bins = 50, range=[0,50], alpha = 0.5, color='r', label ='Answer')
plt.legend()
plt.yscale('log', nonposy= 'clip')
plt.title('Query Lenth Log Histogram by Morph Token')
plt.xlabel('Query Lenth')
plt.ylabel('Number of Queries')


# In[ ]:


plt.figure(figsize = (12, 5))
plt.boxplot([query_sent_len_by_morph, answer_sent_len_by_morph],
           labels = ['Query', 'Answer'])


# In[ ]:


okt.pos('오늘밤은유난히덥구나')


# In[ ]:


query_NVA_token_sentences = list()
answer_NVA_token_sentences = list()

for s in query_sentences:
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            query_NVA_token_sentences.append(token)

for s in answer_sentences:
    temp_token_bucket = list()
    for token, tag in okt.pos(s.replace(' ', '')):
        if tag == 'Noun' or tag == 'Verb' or tag == 'Adjective':
            answer_NVA_token_sentences.append(token)
            
query_NVA_token_sentences = ' '.join(query_NVA_token_sentences)
answer_NVA_token_sentences = ' '.join(answer_NVA_token_sentences)


# In[80]:


from wordcloud import WordCloud

query_wordcloud = WordCloud(font_path= '/Users/hyeongdoson/Downloads/NanumFontSetup_TTF_ALL/' + 'NanumGothic.ttf').generate(query_NVA_token_sentences)

plt.imshow(query_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[82]:


query_wordcloud = WordCloud(font_path = '/Users/hyeongdoson/Downloads/NanumFontSetup_TTF_ALL/' + 'NanumGothic.ttf').generate(answer_NVA_token_sentences)
plt.imshow(query_wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




