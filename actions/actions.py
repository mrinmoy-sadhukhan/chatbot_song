from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

from helper import SongPredcition 
song_pre=SongPredcition()
        

class ActionMusicRecomendation(Action):
    def name(self)  ->  Text:
        return "custom_musicsong"

    def run (self,dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text,Any]) -> List[Dict[Text,Any]]:
       #music = tracker.latest_message.get('music') 
       music = tracker.get_slot("music")
       print(music)  
       dispatcher.utter_message(music)
       song_handle=song_pre.recommender([{'title': music}])
       for i  in range(len(song_handle)):
           print(song_handle[i]['title']+" "+ song_handle[i]['first_artist']+" "+"https://open.spotify.com/track/"+song_handle[i]['id'])
           ##https://open.spotify.com/track/0mUrj2r2vS9x1dqyVoNsB2 
           ###0mUrj2r2vS9x1dqyVoNsB2 can be found id of song database
       dispatcher.utter_message("Here is your Recomended songs with spotify direct link to play") 
       for i  in range(len(song_handle)):
           dispatcher.utter_message(str(i))
           dispatcher.utter_message(str(song_handle[i]['title'])+" "+ str(song_handle[i]['first_artist'])+" "+"https://open.spotify.com/track/"+str(song_handle[i]['id'])+" "+"\n")   
       dispatcher.utter_message("Thank You")
       return []
    
        

