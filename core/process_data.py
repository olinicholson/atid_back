import pandas as pd
from datetime import datetime, timedelta, timezone
from collections import Counter
import pytz
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import numpy as np

class ProcessData:
    def __init__(self, mentions_file:str, tweets_file:str):
        """Initialize the ProcessData class with file paths and load necessary resources."""
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('spanish'))
        self.classifier = classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
        self.mentions_df = pd.read_csv(mentions_file)
        self.tweets_df = pd.read_csv(tweets_file)

    def preprocess_data(self) -> None:
        """Preprocess the data by converting date columns to datetime objects."""
        self.mentions_df['Created At'] = pd.to_datetime(self.mentions_df['Created At'], format='%a %b %d %H:%M:%S %z %Y')
        self.tweets_df['Created At'] = pd.to_datetime(self.tweets_df['Created At'], format='%a %b %d %H:%M:%S %z %Y')
    
    def filter_recent_data(self, days:int=30) -> tuple:
        """Filter the data for the last 'days' days.
        Args:
            days (int): Number of days to look back from today.
        Returns:
            tuple: Two DataFrames containing mentions and tweets from the last 'days' days.
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        mentions_last_days = self.mentions_df[(self.mentions_df['Created At'] >= start_date) & (self.mentions_df['Created At'] <= end_date)]
        tweets_last_days = self.tweets_df[(self.tweets_df['Created At'] >= start_date) & (self.tweets_df['Created At'] <= end_date)]

        previous_start_date = start_date - timedelta(days=days)
        mentions_last_period = self.mentions_df[(self.mentions_df['Created At'] >= previous_start_date) & (self.mentions_df['Created At'] <= start_date)]
        tweets_last_period = self.tweets_df[(self.tweets_df['Created At'] >= previous_start_date) & (self.tweets_df['Created At'] <= start_date)]

        return mentions_last_days, tweets_last_days, mentions_last_period, tweets_last_period

    def calculate_mentions_total(self, mentions: pd.DataFrame) -> int:
        """Filter the data for the last 'days' days.
        Args:
            mentions (pd.Dataframe): Dataframe that contains the data.
        Returns:
            int: Number of mentions.
        """
        return len(mentions)
    
    def calculate_engagement(self, tweets: pd.DataFrame) -> np.float64:
        """Filter the data for the last 'days' days.
        Args:
            tweets (dataframe): Dataframe containing the tweets.
        Returns:
            tuple: Two DataFrames containing mentions and tweets from the last 'days' days.
        """
        total_engagement = (tweets['Retweets'] + tweets['Likes'] + tweets['Quote_count'] + tweets['Reply_count']).sum()
        total_views = tweets['View_count'].sum()
        return (total_engagement / total_views) * 100 if total_views > 0 else 0
    
    def calculate_reach_total(self, tweets: pd.DataFrame) -> np.int64:
        """Filter the data for the last 'days' days.
        Args:
            tweets (dataframe): Dataframe containing the tweets.
        Returns:
            int: Number of total reach.
        """
        return tweets['View_count'].sum()
    
    def most_frequent_words(self, mentions: pd.DataFrame) -> list:
        """Filter the data for the last 'days' days.
        Args:
            mentions (dataframe): Dataframe containing the mentions.
        Returns:
            list: list of dictionaries containing word, count and sentiment.
        """
        all_text = ' '.join(mentions['Text'].fillna(''))
        usual_words = ['pasa', 'puedo', 'hace', 'desde', 'manga', 'puede', 'porque', 
                    'tiene', 'tengo', 'estoy', 'esta', 'está', 'estás', 'estamos', 
                    'están', 'ser', 'soy', 'eres', 'es', 'son', 'fue', 'fui', 
                    'fueron', 'será', 'sería', 'veces', 'tener']
        
        # Primero procesamos las palabras como antes
        words = [
            word.lower()
            for word in all_text.split()
            if (word.lower() not in self.stop_words and 
                len(word) > 3 and 
                "telecentro" not in word.lower() and 
                word.lower() not in usual_words)
        ]
        
        word_counts = Counter(words)
        most_common_words = word_counts.most_common(10)
        
        # Ahora analizamos el sentimiento de cada palabra
        result = []
        for word, count in most_common_words:
            # Analizamos el sentimiento de la palabra
            sentiment_result = self.classifier(word)
            bad_sentiment = sentiment_result[0]['label'] in ['1 star', '2 stars']
            good_sentiment = sentiment_result[0]['label'] in ['4 star', '5 stars']
            
            if sentiment_result[0]['score'] > 0.5 and bad_sentiment:
                sentiment = -1
            elif sentiment_result[0]['score'] > 0.5 and good_sentiment:
                sentiment = 1
            else:
                sentiment = 0
            
            result.append({
                'word': word,
                'count': count,
                'sentiment': sentiment
            })
        
        return result

    def analyze_sentiment(self, mentions: pd.DataFrame) -> dict:
        """Filter the data for the last 'days' days.
        Args:
            tweets (dataframe): Dataframe containing the tweets.
        Returns:
            dict: containing porcentages of sentiment.
        """
        sentiments = []
        for text in mentions['Text']:
            result = self.classifier(text)
            bad_sentiment = result[0]['label'] == '1 star' or result[0]['label'] == '2 stars'
            good_sentiment = result[0]['label'] == '4 star' or result[0]['label'] == '5 stars'
            if result[0]['score'] > 0.5 and bad_sentiment:
                sentiments.append(-1)
            elif result[0]['score'] > 0.5 and good_sentiment:
                sentiments.append(1)
            else:
                sentiments.append(0)
        positivos = sentiments.count(1)
        negativos = sentiments.count(-1)
        total_validos = len(sentiments)
        porcentaje_positivo = (positivos / total_validos) * 100 if total_validos > 0 else 0
        porcentaje_negativo = (negativos / total_validos) * 100 if total_validos > 0 else 0
        sentiments = {'positivos': porcentaje_positivo, 'negativos': porcentaje_negativo, 'neutros': 100 - (porcentaje_positivo + porcentaje_negativo)}
        return sentiments
    def get_stats(self, days=30) -> dict:
        self.preprocess_data()
        # Filter the data for the last 'days' days
        mentions_last_days, tweets_last_days, mentions_last_period, tweets_last_period = self.filter_recent_data(days=days)
        total_mentions = self.calculate_mentions_total(mentions_last_days)
        total_engagement = self.calculate_engagement(tweets_last_days)
        total_reach = self.calculate_reach_total(tweets_last_days)
        most_frequent = self.most_frequent_words(mentions_last_period)
        sentiment = self.analyze_sentiment(mentions_last_days)

        # Get the stats for the previous period
        total_mentions_prev = self.calculate_mentions_total(mentions_last_period)
        total_engagement_prev = self.calculate_engagement(tweets_last_period)
        total_reach_prev = self.calculate_reach_total(tweets_last_period)
        most_frequent_prev = self.most_frequent_words(mentions_last_period)
        sentiment_prev = self.analyze_sentiment(mentions_last_period)

        # Prepare the data for the last 'days' and the previous period
        data_last_days = {
            'total_mentions': total_mentions,
            'total_engagement': total_engagement,
            'total_reach': total_reach,
            'most_frequent_words': most_frequent,
            'sentiment': sentiment}
        data_last_period = {
            'total_mentions': total_mentions_prev,
            'total_engagement': total_engagement_prev,
            'total_reach': total_reach_prev,
            'most_frequent_words': most_frequent_prev,
            'sentiment': sentiment_prev}
        return {'last_days': data_last_days, 'previous_period': data_last_period}

    def compare_periods(self) -> dict:
        """Compare the number of mentions in the current period with the previous period.
        Args:
            current_mentions (pd.DataFrame): DataFrame containing mentions from the current period.
            previous_mentions (pd.DataFrame): DataFrame containing mentions from the previous period.
        Returns:
            dict: containing the difference and percentage change.
        """
        data = self.get_stats()
        data_last_days = data['last_days']
        data_last_period = data['previous_period']
        current_mentions = data_last_days['total_mentions']
        previous_mentions = data_last_period['total_mentions']
        difference_mentions = current_mentions - previous_mentions
        percentage_change_mentions = (difference_mentions / previous_mentions * 100) if previous_mentions != 0 else 0

        current_engagement = data_last_days['total_engagement']
        previous_engagement = data_last_period['total_engagement']
        difference_engagement = current_engagement - previous_engagement
        percentage_change_engagement = (difference_engagement / previous_engagement * 100) if previous_engagement != 0 else 0

        current_reach = data_last_days['total_reach']
        previous_reach = data_last_period['total_reach']
        difference_reach = current_reach - previous_reach
        percentage_change_reach = (difference_reach / previous_reach * 100) if previous_reach != 0 else 0

        sentiment_last_days = data_last_days['sentiment']
        positive_sentiment_last_days = sentiment_last_days['positivos']
        negative_sentiment_last_days = sentiment_last_days['negativos']
        neutral_sentiment_last_days = sentiment_last_days['neutros']
        sentiment_last_period = data_last_period['sentiment']
        positive_sentiment_last_period = sentiment_last_period['positivos']
        negative_sentiment_last_period = sentiment_last_period['negativos']
        neutral_sentiment_last_period = sentiment_last_period['neutros']
        sentiment_change = {
            'positive': positive_sentiment_last_days - positive_sentiment_last_period,
            'negative': negative_sentiment_last_days - negative_sentiment_last_period,
            'neutral': neutral_sentiment_last_days - neutral_sentiment_last_period
        }
        percentage_change_sentiment = {
            'positive': (sentiment_change['positive'] / positive_sentiment_last_period * 100) if positive_sentiment_last_period != 0 else 0,
            'negative': (sentiment_change['negative'] / negative_sentiment_last_period * 100) if negative_sentiment_last_period != 0 else 0,
            'neutral': (sentiment_change['neutral'] / neutral_sentiment_last_period * 100) if neutral_sentiment_last_period != 0 else 0
        }
        return {'mentions': percentage_change_mentions, 'engagement': percentage_change_engagement, 'reach': percentage_change_reach,  'sentiment': percentage_change_sentiment}
    def get_last_mentions(self) -> pd.DataFrame:
        """Get the last mentions from the mentions DataFrame.
        Returns:
            pd.DataFrame: DataFrame containing the last mentions with sentiment.
        """
        self.preprocess_data()
        tweets_list = []
        tweets = self.mentions_df.sort_values(by='Created At', ascending=False).head(5)
        for i in tweets.index:
            interacciones = (
                tweets['Retweets'][i] + 
                tweets['Likes'][i] + 
                tweets['Quote_count'][i] + 
                tweets['Reply_count'][i]
            )
            
            # Analizar sentimiento
            result = self.classifier(tweets['Text'][i])
            bad_sentiment = result[0]['label'] == '1 star' or result[0]['label'] == '2 stars'
            good_sentiment = result[0]['label'] == '4 star' or result[0]['label'] == '5 stars'
            if result[0]['score'] > 0.5 and bad_sentiment:
                sentiment = -1
            elif result[0]['score'] > 0.5 and good_sentiment:
                sentiment = 1
            else:
                sentiment = 0

            # Agregar a la lista
            tweets_list.append({
                'User Name': tweets['Username'][i],
                'Text': tweets['Text'][i],
                'Created At': tweets['Created At'][i],
                'Interactions': int(interacciones),
                'Sentiment': sentiment,
            })

        return tweets_list