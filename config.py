import os 
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
Mongo_URI = os.getenv('MONGO_URI')
client = MongoClient(Mongo_URI)
db=client['thoracic_surgery_db']