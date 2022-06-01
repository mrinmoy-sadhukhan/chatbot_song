import numpy as np
import pandas as pd
import nltk
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

class SongPredcition:

    def __init__(self):

        
        self.credentials = json.load(open('authorization.json'))
        self.client_id = self.credentials['client_id']
        self.client_secret = self.credentials['client_secret']
        self.client_credentials_manager = SpotifyClientCredentials(client_id=self.client_id,client_secret=self.client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager=self.client_credentials_manager)
        self.number_cols = ['valence', 'acousticness', 'danceability', 'duration_ms', 'energy', 
        'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'tempo','time_signature']
        
        self.song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=20, 
                                   verbose=False, n_jobs=4))
                                 ], verbose=False)

        data=pd.read_csv("playlist_0.csv")
        data.drop(['num_segments','num_sections','num_bars','all_artists'], inplace=True, axis=1)
        #self.data=self.data.drop('num_sections', inplace=True, axis=1)
        #self.data=self.data.drop('num_bars', inplace=True, axis=1)
        #self.data=self.data.drop('all_artists',inplace=True,axis=1)
        #self.data1=self.data.drop('id',inplace=True,axis=1)
        self.data1=data

        self.X = data.select_dtypes(np.number)
        #number_cols = list(X.columns)
        self.song_cluster_pipeline.fit(self.X)
        #self.song_cluster_labels = self.song_cluster_pipeline.predict(self.X)
        #self.data['cluster_label'] = self.song_cluster_labels
        #print("test")


    def find_song(self,name):
        song_data = defaultdict()
        results = self.sp.search(q= 'track: {} '.format(name), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = self.sp.audio_features(track_id)[0]

        song_data['title'] = [name]
        #song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]
    
        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)
    
    def get_song_data(self, song , spotify_data):
    
        try:
            song_data = spotify_data[(spotify_data['title'] == song['title']) 
                                ].iloc[0]
            return song_data
    
        except IndexError:
            return self.find_song(song['title'])
    
    def get_mean_vector(self,song_list, spotify_data):
    
        song_vectors = []
    
        for song in song_list:
            song_data = self.get_song_data(song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database'.format(song['title']))
                continue
            song_vector = song_data[self.number_cols].values
            song_vectors.append(song_vector)  
    
        song_matrix = np.array(list(song_vectors))
        return np.mean(song_matrix, axis=0)

    def flatten_dict_list(self,dict_list):
            
        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []
    
        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)
            
        return flattened_dict

    def recommender(self, song_list, n_songs=10): ##
        spotify_data=self.data1   
        #print(spotify_data)
        metadata_cols = ['title', 'first_artist','id']
        song_dict = self.flatten_dict_list(song_list)
    
        song_center = self.get_mean_vector(song_list, spotify_data)
        scaler = self.song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[self.number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])
    
        rec_songs = spotify_data.iloc[index]
        rec_songs = rec_songs[~rec_songs['title'].isin(song_dict['title'])]
        return rec_songs[metadata_cols].to_dict(orient='records')

   

#if __name__=="__main__":
#    print("from main")
#    recomendation=SongPredcition()
#    print(recomendation.recommender([{'title': 'Chhoti Si Aasha'}]))
    

