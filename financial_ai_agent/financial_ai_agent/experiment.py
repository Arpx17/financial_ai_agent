from pymongo import MongoClient

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["news_db"]
collection = db["articles"]

count = 0
for per in collection.find():
    count+=1

print(count)
print(per)