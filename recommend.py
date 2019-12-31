import pandas as pd
from user_item_based import *
from content_based import *

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')


uib = ArticleRecommender(10,df)
cb = ContentBasedRecommendations(df_content)

print('###########################################################################################################')
print('Recommendation for User with id 20 - User-User Based Collaborative Filtering')

print(uib.make_recommendations(20))

print('###########################################################################################################')
print('Recommendation for User with id 5149 - Rank Based Recomendation')

print(uib.make_recommendations(5149))

print('###########################################################################################################')
print('Recommendations for a user who only has interacted with Article id 1427 -  Content Based Recommendations')

print(cb.get_recommendations_doc2vec(df[df.article_id == 1427.0].title.values[0]))


