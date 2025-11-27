from calendar import c
from email.mime import base
from multiprocessing import process
from torchreid.utils import FeatureExtractor
import cv2
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import datetime
import torch
import itertools
import glob
import os
import time
from pprint import pprint
import json
import pytz
import pymongo
import shutil 
from collections import Counter
import autonomoShopperOpenEventsHandler
# from productPersonSlackNotification import slack_notification

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["CVAutonomo"]
myevents = mydb["ProductEvents"]
with open('/home/orange/Production/RecommendationServices/config.json', 'r') as f:
    config = json.load(f)

print(config)
extractor = FeatureExtractor(
            model_name=config["models"]["osNetModelName"],
            model_path=config["models"]["osNetModelPath"],
            device='cuda'
        )
def connectMongoDB(tDate):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    myevents = mydb["ProductEvents"]
    # eventData = datetime.datetime.now().strftime("%Y%m%d")
    docs = myevents.find({'videoProcessed': True, "personPairing": False, "EventDate": { '$gt': "20250517" }})
    my_data = []
    for doc in docs:
        my_data.append(doc)
    myclient.close()
    return my_data

def updateMongoDB_None(eventID):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    myevents = mydb["ProductEvents"]
    myquery = { "EventId": eventID}
    processedTime = time.time() - st_time
    newvalues = { "$set": { "personPairing": None, "processedTime": processedTime} }
    myevents.update_one(myquery, newvalues)
    myclient.close()

def updateMongoDB_Recommended(eventID, totalPersonRecommended, entry_ids_list, processedTime):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    myevents = mydb["ProductEvents"]
    myquery = { "EventId": eventID}
    # processedTime = time.time() - st_time
    newvalues = { "$set": { "personPairing": totalPersonRecommended, 'entryIds':entry_ids_list, "processedTime": processedTime} }
    myevents.update_one(myquery, newvalues)
    myclient.close()



def master_images_path(base_path, camera_id, day):
    """Return the Master Image Folder path for entry images
        Parameters:
        ----------
        base_path : string
                Base Path where we are storing all the images and Hawkeye *(anhoody001)
        camera_id : string
                Camera that we are using for storing the files inside that path
        
        Returns:
        -------
        master_img_path : string 
            returns the Master image path that is created based on datetime and camera no.
    """
    master_event_path = os.path.join(base_path,"master_event")
    exit_camera_id = camera_id
    if not os.path.exists(master_event_path):
        os.makedirs(master_event_path)
    master_entry_camera_path = os.path.join(master_event_path, exit_camera_id)
    if not os.path.exists(master_entry_camera_path):
        os.makedirs(master_entry_camera_path)
    # day = datetime.datetime.now().strftime('%Y%m%d')
    master_img_path = os.path.join(master_entry_camera_path, day)
    if not os.path.exists(master_img_path):
        os.makedirs(master_img_path)
    return master_img_path

# camera_no = config["entryCameraNo"]
base_path= config["basePath"]
base_images_folder = config["productEventImgsPath"]
if not os.path.exists(base_images_folder):
    os.makedirs(base_images_folder)


def eventTimeUTC(eventTime):
    """Convert the Local time to UTC
        Parameters:
        ----------
        eventTime : string
                The Local event time that is taken as input string
        
        Returns:
        -------
        tm_utc : string 
                returns the utc time format for the input local time.
    """
    tm = datetime.datetime.strptime(str(eventTime), "%Y-%m-%dT%H%M%S")
    tm_utc = tm.astimezone(pytz.UTC)
    try:
        tm_utc = tm_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        tm_utc = tm_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return tm_utc

def calculate_target_feature(target_box):
    """ Extract the features from the input image
        Parameters:
        ----------
        target_box : numpy array
                The Image that is given as input to extract features
        
        Returns:
        -------
        target_np_array : numpy array 
                returns the features extracted from the input image.
    """
    features = extractor(target_box)
    target_np_array = features.cpu().detach().numpy()
    return target_np_array


def eventFeatureExtraction(eventID, camera):
    """ Extract the features for all the images extracted from the Event
        Parameters:
        ----------
        eventID : string
                The Uniqie ID that is used when we are extracting the images from video
        camera : string
            The camera from which the event has occurred. 
        
        Returns:
        -------
        event_filenames : list 
                returns all the list of filenames from which we extracted features.
        event_images_features : list 
                returns all the list of features extracted from the images.
    """
    path =  str(eventID)+"_compressed_" + str(camera)
    folderPath = os.path.join(base_images_folder, path)
    images = glob.glob(str(folderPath)+"/*.jpg")
    images = [x.replace("\\", "/") for x in images]
    # print(folderPath)
    # print(len(images))
    event_images_features = []
    event_filenames = []
    count = 0
    """NOTE: Check Number of Images in folder"""
    for image in images:
        count += 1
        if count %2 == 0:
            filename = image.split("/")[-1]
            fileid = filename.split("_")[0]
            event_filenames.append(fileid)
            features = calculate_target_feature(image)
            event_images_features.append(features)
    # shutil.rmtree(folderPath)
    return event_filenames, event_images_features


def extractEntryImagesFeaturesDatabase(entry_ids_list):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    mycol = mydb["Entry"]
    # for entry_id in entry_ids_list:
    find_query = {'engAssignedID': {'$in': entry_ids_list}}
    entry_infos = list(mycol.find(find_query))
    temp = []
    for i in range(len(entry_infos)):
        entry_imgs = entry_infos[i]["filename"]
    entry_filenames, entry_images_features = [], []
    for entry_info in entry_infos:
        id, filenames = entry_info["engAssignedID"], entry_info["filename"]
        if not type(filenames) == str:
            for filename in filenames:
                entry_filenames.append(id)
                features = calculate_target_feature(filename)
                entry_images_features.append(features)
        else:
            entry_filenames.append(id)
            features = calculate_target_feature(filenames)
            entry_images_features.append(features)
    myclient.close()
    # print(entry_images_features)
    return entry_filenames, entry_images_features

def extractEntryImagesFeatures(entry_ids_list, eventDate):
    """ Extract the features for all the entry images based on the entry id list
        Parameters:
        ----------
        entry_ids_list : list
                All the Entry id in a list that were active while the event happened.
        Returns:
        -------
        entry_filenames : list 
                returns all the list of entry filenames from which we extracted features.
        entry_images_features : list 
                returns all the list of features extracted from the entry images.
    """
    day = eventDate
    master_entry_imgs_path = master_images_path(base_path, camera_no, day)
    images = glob.glob(master_entry_imgs_path+"/*.jpg")
    images = [x.replace("\\", "/") for x in images]
    print("len of entry images: ", len(images))
    entry_images_features = []
    entry_filenames = []
    for image in images:
        if any(entry_id in image for entry_id in entry_ids_list):
            # print("Filename: ", image)
            filename = image.split("/")[-1]
            fileid = filename.split("_")[0]
            entry_filenames.append(fileid)
            features = calculate_target_feature(image)
            entry_images_features.append(features)

    return entry_filenames, entry_images_features


def RecommentationList(entry_ids_list, eventID, camera, eventDate):
    """ Extract the features for all the entry images based on the entry id list
        Parameters:
        ----------
        entry_ids_list : list
                All the Entry id in a list that were active while the event happened.
        Returns:
        -------
        entry_filenames : list 
                returns all the list of entry filenames from which we extracted features.
        entry_images_features : list 
                returns all the list of features extracted from the entry images.
    """
    entry_filenames, entry_images_features = extractEntryImagesFeaturesDatabase(entry_ids_list)
    event_filenames, event_images_features = eventFeatureExtraction(eventID, camera)
    # print(len(entry_filenames))
    print("EventID : ", eventID)
    print("Features ", len(event_images_features), len(entry_images_features))
    if len(event_images_features) == 0 or len(entry_images_features) == 0:
        personRecommened = {}
        top_10_ids = []
        return top_10_ids, personRecommened
    complete_list = []
    for i in range(0, len(event_filenames)):
        for j in range(0, len(entry_filenames)):
            temp = []
            target_embedding = event_images_features[i]
            entry_image = entry_images_features[j]
            matching_results =  euclidean_distances(target_embedding, entry_image)
            # print("Matching Results: ", matching_results)
            temp.append(event_filenames[i])
            temp.append(entry_filenames[j])
            matching_value = round((100-matching_results[0][0]), 2)
            temp.append(matching_value)
            complete_list.append(temp)
    df = pd.DataFrame(complete_list, columns=["event_id", "entry_id", "conf"])
    temp = df[df["conf"] > 70]
    temp = temp.sort_values(by=['conf'], ascending=False)
    top_10_ids = list(temp["entry_id"])
    top_10_conf = list(temp["conf"])
    top_10_ids = top_10_ids[:10]
    top_10_conf = top_10_conf[:10]
    persons_recommended = {}
    for id in range(len(top_10_ids)):
        if not top_10_ids[id] in persons_recommended.keys():
            persons_recommended[top_10_ids[id]] = top_10_conf[id]
    return top_10_ids, persons_recommended

def minChange(prev_min):
    time_now = datetime.datetime.now()
    current_time = time_now.strftime("%Y%m%d_%H%M%S.%f")
    current_min = current_time.split(".")[0][-4:-2]
    if prev_min == current_min:
        return False
    else:
        return True

time_prev = datetime.datetime.now()
prev_time = time_prev.strftime("%Y%m%d_%H%M%S.%f")
prev_min = prev_time.split(".")[0][-4:-2]
while True:
    try:
        minChangeFlag = minChange(prev_min)
        # if minChangeFlag == False:
        #     continue
        temp_time_prev = datetime.datetime.now()
        temp_prev_time = temp_time_prev.strftime("%Y%m%d_%H%M%S.%f")
        prev_min = temp_prev_time.split(".")[0][-4:-2]
        print(prev_min)
        todays_date = datetime.datetime.now().strftime('%Y%m%d')
        tDate = todays_date.replace('-', '')
        # for doc in myevents.find({'blurringCompleted': True, "personPairing": False, 'EventDate':todays_date}):
        docs = connectMongoDB(tDate)
        for doc in docs:
            if doc["personsPresent"] == 0:
                continue
            st_time = time.time()
            eventID = doc['EventId']
            print(eventID)
            eventDate = str(doc["EventDate"])
            eventTime = str(doc['EventTime'])
            eventTime = str(eventTime).replace(' ', "T")
            cameras = doc["CamsAssociated"]
            eventTime = str(eventTime).replace(':', "")
            timing = eventTimeUTC(eventTime)
            entry_ids_list = doc["entryIds"]
            if entry_ids_list is None or len(entry_ids_list) == 0 :
                response_json = autonomoShopperOpenEventsHandler.getOpenShoppers(timing)
                
                # print("Response:", response_json)
                entry_ids_list = [temp["id"] for temp in response_json]
            # entry_ids_list = ["630c5f4c61b23511d46a9e1f"]
            totalPersonRecommended = {}
            print("Entry Ids: ",entry_ids_list)
            if entry_ids_list is None or len(entry_ids_list) == 0:
                print("No Entry IDs Found")
                # final_recommendation = {}
                updateMongoDB_None(eventID)
                output_response = {
                "eventID": eventID,
                "activityList": totalPersonRecommended
                    }
                autonomoShopperOpenEventsHandler.assignRecommendedShoppers(output_response)
            else:
                for camera in cameras:
                    ids, personRecommened = RecommentationList(entry_ids_list, eventID, camera, eventDate)
                    path = str(eventID) + "_" + str(camera)
                    if len(personRecommened) > 0:
                        ids_count = Counter(ids)
                        ids_count = dict(ids_count)
                        temp = [k for k,v in ids_count.items() if int(v) >= 6]
                        if len(temp) == 0:
                            # max_value = max(ids_count, key=ids_count.get)
                            max_value = max(list(ids_count.values()))
                            recommended_ids = [k for k,v in ids_count.items() if int(v) >= max_value]
                            final_recommendation = {}
                            for recom_id in recommended_ids:
                                if recom_id in totalPersonRecommended.keys():
                                    if personRecommened[recom_id] > totalPersonRecommended[recom_id]:
                                        totalPersonRecommended[recom_id] = personRecommened[recom_id]
                                else:
                                    totalPersonRecommended[recom_id] = personRecommened[recom_id]
                        else:
                            final_recommendation = {}
                            for final_id in temp:
                                if final_id in totalPersonRecommended.keys():
                                    if personRecommened[final_id] > totalPersonRecommended[final_id]:
                                        totalPersonRecommended[final_id] = personRecommened[final_id]
                                else:
                                    totalPersonRecommended[final_id] = personRecommened[final_id]
                    
                    temp =  str(eventID)+"_compressed_" + str(camera)
                    folderPath = os.path.join(base_images_folder, temp)
                    if os.path.exists(folderPath):
                        shutil.rmtree(folderPath)
                    # compressedPath =  str(path) + "_compressed"
                    # compressedFolderPath = os.path.join(base_images_folder, compressedPath)
                    print("Compress Path: ", folderPath)
                    # if os.path.exists(compressedFolderPath):
                    #     shutil.rmtree(compressedFolderPath)
            print("Total Person Recommended: ", totalPersonRecommended)
            if len(totalPersonRecommended) > 0:
                processedTime = time.time() - st_time
                output_response = {
                    "eventID": eventID,
                    "activityList": totalPersonRecommended
                }
                print(output_response)
                autonomoShopperOpenEventsHandler.assignRecommendedShoppers(output_response)
                print(f"Person Recommended for {eventID} are {totalPersonRecommended}")
                updateMongoDB_Recommended(eventID, totalPersonRecommended, entry_ids_list, processedTime)
            else:
                processedTime = time.time() - st_time
                output_response = {
                    "eventID": eventID,
                    "activityList": totalPersonRecommended
                }
                print(output_response)
                autonomoShopperOpenEventsHandler.assignRecommendedShoppers(output_response)
                print(f"Person Recommended for {eventID} are {totalPersonRecommended}")
                # slack_notification(f"NoEntryFeatures in {eventID} for Product Person Pairing")
                updateMongoDB_Recommended(eventID, "NoEntryFeatures", entry_ids_list, processedTime)

    except Exception as e:
        print("Exception Occurred in Product Person Pairing: ", e) 
        continue


