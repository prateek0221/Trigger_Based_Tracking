# from detect_oops import FaceBlur
from personExtraction import PersonDetection
import pymongo
import os
import cv2
import json
import datetime

with open('/home/orange/Production/RecommendationServices/config.json', 'r') as f:
    config = json.load(f)

asset_dir = config["basePath"]
# eventDate = datetime.datetime.now().strftime("%Y%m%d")

def connectMongoDB(tDate):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    myevents = mydb["ProductEvents"]
    eventDate = datetime.datetime.now().strftime("%Y%m%d")
    # print("EventData:", str(eventDate))
    docs = myevents.find({"videoRetrieved":True, "videoProcessed": False, "EventDate": { '$gt': "20250617" }})
    my_data = []
    for doc in docs:
        my_data.append(doc)
    # print(my_data)
    myclient.close()
    # print(my_data[-1])
    return my_data

def updateMongoDB(eventID, persons_present):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    myevents = mydb["ProductEvents"]
    myquery = {"EventId": eventID}
    newvalues = {"$set": {"videoProcessed": True, "personsPresent": persons_present}}
    myevents.update_one(myquery, newvalues)

def minChange(prev_min):
    time_now = datetime.datetime.now()
    current_time = time_now.strftime("%Y%m%d_%H%M%S.%f")
    current_min = current_time.split(".")[0][-4:-2]
    if prev_min == current_min:
        return False
    else:
        return True

def main():
    time_prev = datetime.datetime.now()
    prev_time = time_prev.strftime("%Y%m%d_%H%M%S.%f")
    prev_min = prev_time.split(".")[0][-4:-2]
    person_extraction = PersonDetection()
    while True:
        # minChangeFlag = minChange(prev_min)
        # if minChangeFlag == False:
        #     continue
        temp_time_prev = datetime.datetime.now()
        temp_prev_time = temp_time_prev.strftime("%Y%m%d_%H%M%S.%f")
        prev_min = temp_prev_time.split(".")[0][-4:-2]
        todays_date = datetime.datetime.now().strftime('%Y%m%d')
        tDate = todays_date.replace('-', '')
        # print()
        docs = connectMongoDB(tDate)
        for doc in docs:
            try:
                print("Event ID: ", doc["EventId"])
                cameras = doc["CamsAssociated"]
                filename = str(doc["EventId"])+".mp4"
                eventID = str(doc["EventId"])
                eventDate = datetime.datetime.strptime(doc["EventDate"], "%Y%m%d")
                eventDate = eventDate.strftime("%Y-%m-%d")
                productPickupPosition = doc['productPickupPosition']
                todays_date = datetime.datetime.now().strftime('%Y-%m-%d')
                yesterday_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
                if eventDate != str(todays_date) and eventDate != str(yesterday_date) and eventDate != (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d"):
                    continue
                print("Event ID: ", doc["EventId"])
                print("event Date: ", eventDate)
                compressed_filename = str(doc["EventId"])+"_compressed"+".mp4"
                total_persons = 0
                folder_eventDate = datetime.datetime.now().strftime("%Y-%m-%d")
                for camera in cameras:
                    product_event_dir = '/productEventsNAS/'+str(eventDate) + "/" +str(camera)+ "/"
                    # original_event = asset_dir + product_event_dir + filename
                    # blurred_event = asset_dir + product_event_dir + blurred_filename
                    compressed_event = asset_dir + product_event_dir + compressed_filename
                    print
                    personCount = person_extraction.person_detector(compressed_event, camera)
                    if total_persons < personCount:
                        total_persons = personCount
                print(total_persons)
                updateMongoDB(eventID, total_persons)
                print("Updated the Database for the Event ID: ", eventID)
            except Exception as e:
                print("Exception Occured: ", e)
                continue
                
            # cv2.destroyAllWindows()

 
if __name__ == "__main__":
    main()

# /data/store/assets/anhoody001/eventvideosnas/20230904/D15/compressed_64f51358a6def30a26c72b16.mp4
# /data/store/assets/anhoody001/eventvidesonas/20230904/D15/compressed_64f51358a6def30a26c72b16.mp4