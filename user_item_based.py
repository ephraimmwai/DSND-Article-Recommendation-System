import pandas as pd
import numpy as np

class ArticleRecommender:

	def __init__(self, n, df):

		##Check dataframe of format: article_id,title,email
		del df['Unnamed: 0']
		if df.columns.tolist().sort() != ['article_id', 'title', 'email'].sort():
			raise ValueError('The dataframe columns should be article_id, title and email')
		else:
			df['user_id'] = self.email_mapper(df['email'])
			del df['email']

		self.df = df
		self.n = n

		self.user_item = self.create_user_item_matrix()

		# self.neighb = None
		# self.user_id = user_id

	def df_clean(self):
		return self.df


	#Encode email class to digits
	def email_mapper(self, df_email_col):
	    '''
	    INPUT:
	    df_email_col - email column name
	    
	    OUTPUT:
	    email_encoded - encoded email column
	    '''
	    coded_dict = dict()
	    cter = 1
	    email_encoded = []
	    
	    for val in df_email_col:
	        if val not in coded_dict:
	            coded_dict[val] = cter
	            cter+=1
	        
	        email_encoded.append(coded_dict[val])
	    return email_encoded


	def new_users_articles(self):
	    '''   
	    OUTPUT:
	    top_articles - (list) A list of the top 'n' article titles 
	    
	    '''
	    top_articles = list(self.df.groupby(['article_id','title']).size().reset_index(name='count').sort_values(by='count', ascending = False).article_id.head(self.n))    
	    
	    return top_articles # Return the top article titles from df (not df_content)

	def create_user_item_matrix(self):
	    '''
	    OUTPUT:
	    user_item - user item matrix 
	    
	    Description:
	    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
	    an article and a 0 otherwise
	    '''
	    user_item = self.df.drop_duplicates().groupby(['user_id', 'article_id']).size().unstack().fillna(0).astype('int')
	    
	    return user_item # return the user_item matrix 

	def get_article_names(self, article_ids):
	    '''
	    INPUT:
	    article_ids - (list) a list of article ids
	    
	    OUTPUT:
	    article_names - (list) a list of article names associated with the list of article ids 
	                    (this is identified by the title column)
	    '''

	    article_ids = list(map(int, map(float,article_ids)))
	    
	    article_names = self.df[self.df['article_id'].isin(article_ids)]['title'].drop_duplicates().values.tolist()
	    
	    # Return the article names associated with list of article ids
	    return article_names 


	def get_user_articles(self, user_id):
	    '''
	    INPUT:
	    user_id - (int) a user id
	    
	    OUTPUT:
	    article_ids - (list) a list of the article ids seen by the user
	    article_names - (list) a list of article names associated with the list of article ids 
	                    (this is identified by the doc_full_name column in df_content)
	    
	    Description:
	    Provides a list of the article_ids and article titles that have been seen by a user
	    '''
	    # Your code here
	    article_ids = list(self.user_item.loc[user_id][self.user_item.loc[user_id] == 1].index.values.astype('float').astype('str'))  
	    
	    return article_ids


	def get_top_sorted_users(self, user_id):
	    '''
	    INPUT:
	    user_id - (int)
	    df - (pandas dataframe) df as defined at the top of the notebook 
	    user_item - (pandas dataframe) matrix of users by articles: 
	            1's when a user has interacted with an article, 0 otherwise
	    
	            
	    OUTPUT:
	    neighbors_df - (pandas dataframe) a dataframe with:
	                    neighbor_id - is a neighbor user_id
	                    similarity - measure of the similarity of each user to the provided user_id
	                    num_interactions - the number of articles viewed by the user - if a u
	                    
	    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
	                    highest of each is higher in the dataframe
	     
	    '''
	    neighbors_df  = pd.DataFrame(columns=['neighbor_id', 'similarity', 'num_interactions'])
	 
	    for neighb in self.user_item.index.values:
	        if neighb != user_id:
	            sim = self.user_item[self.user_item.index == user_id].dot(self.user_item.loc[neighb].transpose()).values[0]
	            num_inter = self.user_item.loc[neighb].values.sum()

	            neighbors_df.loc[neighb] = [neighb, sim, num_inter]
	            
	    neighbors_df[['similarity','num_interactions']]= neighbors_df[['similarity','num_interactions']].astype('int')
	    neighbors_df = neighbors_df.sort_values(by =['similarity', 'neighbor_id'], ascending = [False, True])
	    
	    return neighbors_df


	def user_based_recs(self, user_id):
	    '''
	    INPUT:
	    user_id - (int) a user id you want to make recommendations for
	    
	    OUTPUT:
	    recs - (list) a list of recommendations for the user by article id
	    rec_names - (list) a list of recommendations for the user by article title
	    
	    Description:
	    Loops through the users based on closeness to the input user_id
	    For each user - finds articles the user hasn't seen before and provides them as recs
	    Does this until m recommendations are found
	    
	    Notes:
	    * Choose the users that have the most total article interactions 
	    before choosing those with fewer article interactions.

	    * Choose articles with the articles with the most total interactions 
	    before choosing those with fewer total interactions. 
	   
	    '''
	    df_sorted_usrs = self.get_top_sorted_users(user_id)
	    neighbs = df_sorted_usrs['neighbor_id'].values.tolist()
	    
	    recs = []
	    #articles a user has read
	    user_reads = list(set(self.get_user_articles(user_id)))
	    
	    for neighb in neighbs:

	        neighb_reads = self.get_user_articles(neighb)

	        #Obtain recommendations for each neighbor
	        new_recs = np.setdiff1d(neighb_reads, user_reads, assume_unique=True)

	        # Update recs with new recs
	        recs = list(np.unique(np.concatenate([new_recs, recs], axis=0)))[:self.n]
	        
	    rec_names = self.get_article_names(recs)
	        
	    return recs, rec_names

	def make_recommendations(self, user_id):

		user_reads = list(set(self.get_user_articles(user_id)))

		#check if new user
		if len(user_reads) < 2:
			top_articles = self.new_users_articles()
			recs = np.setdiff1d(top_articles, user_reads, assume_unique=True)
			rec_names = self.get_article_names(recs)
		else:
			recs, rec_names = self.user_based_recs(user_id)

		return recs, rec_names

