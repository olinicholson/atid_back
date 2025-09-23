import csv
import os

class FileManager:
    def __init__(self, file_name: str):
        self.file_name = file_name

    def ensure_file_exists(self, headers: list):
        """Ensures the file exists and has the correct headers."""
        if not os.path.exists(self.file_name):
            with open(self.file_name, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
            print(f"New file created!: {self.file_name} with headers: {headers}")

    def get_existing_ids(self) -> set:
        """Retrieves all existing tweet IDs from the file."""
        existing_ids = set()
        if os.path.exists(self.file_name):
            with open(self.file_name, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    existing_ids.add(row["Id"])  # Assumes 'tweet_id' is a header
        return existing_ids

    def append_tweets(self, tweets: list, headers: list):
        """
        Appends a list of tweets to the file. Ensures the file exists and has the correct headers.
        Filters out tweets whose IDs already exist in the file.
        """
        self.ensure_file_exists(headers)  # Ensure the file exists before appending

        existing_ids = self.get_existing_ids()  # Fetch existing tweet IDs

        # Filter out tweets with duplicate IDs
        new_tweets = [tweet for tweet in tweets if tweet["tweet_id"] not in existing_ids]

        if not new_tweets:
            print(f"No new tweets to append to {self.file_name}")
            return

        # Append only new tweets
        with open(self.file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            for tweet in new_tweets:
                writer.writerow([
                    tweet["tweet_id"],
                    tweet["user_name"],
                    tweet["text"],
                    tweet["created_at"],
                    tweet["retweet_count"],
                    tweet["favorite_count"],
                    tweet["quote_count"],
                    tweet["reply_count"],
                    tweet["view_count"]
                ])
        print(f"Appended {len(new_tweets)} new tweets to {self.file_name}")
