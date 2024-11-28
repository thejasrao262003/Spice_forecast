import os
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv
import certifi
from datetime import datetime

# Load environment variables
load_dotenv()

# MongoDB connection
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("MongoDB URI is not set!")
else:
    # Connect to MongoDB with SSL certificate validation
    client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
    db = client["AgriPredict"]
    collection = db["WhiteSesame"]
    print(f"Connected to database: {db.name}")
    print(f"Using collection: {collection.name}")

    # Update Reported Date field from string or datetime to date
    documents_to_update = collection.find({"Reported Date": {"$exists": True}})

    bulk_updates = []
    for doc in documents_to_update:
        try:
            # Check the type of 'Reported Date'
            reported_date = doc["Reported Date"]

            if isinstance(reported_date, str):
                # If it's a string, parse it to a datetime object
                reported_date_dt = datetime.strptime(reported_date, "%Y-%m-%d")
            elif isinstance(reported_date, datetime):
                # If it's already a datetime object, normalize to date
                reported_date_dt = datetime(reported_date.year, reported_date.month, reported_date.day)
            else:
                # Skip if it's neither a string nor a datetime
                print(f"Unsupported type for 'Reported Date' in document with _id {doc['_id']}: {type(reported_date).__name__}")
                continue

            # Prepare the update query
            bulk_updates.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"Reported Date": reported_date_dt}}
                )
            )
        except Exception as e:
            print(f"Error processing document with _id {doc['_id']}: {e}")

    # Execute bulk update
    if bulk_updates:
        result = collection.bulk_write(bulk_updates)
        print(f"Updated {result.modified_count} documents.")
    else:
        print("No documents found with 'Reported Date' field or no updates required.")
