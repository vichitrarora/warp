from pymongo import MongoClient
import urllib.parse
import re

username = urllib.parse.quote_plus("vichitrarora")
password = urllib.parse.quote_plus("Vishu@133824")  

uri = f"mongodb+srv://{username}:{password}@cluster0.7j18x.mongodb.net/?retryWrites=true&w=majority"

def check_query_executable(shell_query: str) -> int:
    try:
        client = MongoClient(uri)
        db = client["test"]

        match = re.match(r"db\.(\w+)\.find\((.*)\)", shell_query.strip())
        if not match:
            return 0  

        collection_name = match.group(1)
        query_str = match.group(2)

        query_dict = eval(query_str)  

        db[collection_name].count_documents(query_dict)
        return 1  

    except Exception:
        return 0  

shell_query = 'db.users.find({ "name": "John" })'
reward = check_query_executable(shell_query)
print(f"Reward: {reward}")
