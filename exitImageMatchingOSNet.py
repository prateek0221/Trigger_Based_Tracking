from torchreid.utils import FeatureExtractor
import cv2
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import datetime
from datetime import date, timedelta
import pytz
import glob
import os
import time
from pprint import pprint
import pymongo 
import numpy as np
import autonomoShopperOpenEventsHandler
from collections import Counter
import autonomoEntryExitActivityHandler
import shutil

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["CVAutonomo"]
exit_features = mydb["Exit"]
entry_features = mydb["Entry"]

extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path="/home/orange/Production/RecommendationServices/models/model.pth.tar-80",
            device='cuda'
        )


def f(counter, exit_id, exit_time, entry_ids, exitimg_filepath, exit_eng_id, ids, entry_ids_list, exit_ids_list, entry_imgs_path, exit_imgs_path, df):
    
    current_date = date.today().strftime("%Y-%m-%d")
    base_path_value = '/home/orange/assets/atn-bako-001/recommendation_data'
    if not os.path.exists(base_path_value):
        os.makedirs(base_path_value)
    date_folder_path = os.path.join(base_path_value, current_date)
    if not os.path.exists(date_folder_path):
        os.makedirs(date_folder_path)

    folder_name = f'folder_{counter}'
    folder_path = os.path.join(date_folder_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    txt_file_path = os.path.join(folder_path, f'details.txt')
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write("exit_id: " + str(exit_id) + '\n')
        txt_file.write("exit_time: " + str(exit_time) + '\n')
        txt_file.write("entry_ids: " + str(entry_ids) + '\n')
        txt_file.write("exitimg_filepath: " + str(exitimg_filepath) + '\n')
        txt_file.write("exit_eng_id: " + str(exit_eng_id) + '\n')

    entry_imgs_folder = os.path.join(folder_path, 'entry_images')
    os.makedirs(entry_imgs_folder, exist_ok=True)

    # Save entry images
    for file in entry_imgs_path:
        try:
            output_path = os.path.join(entry_imgs_folder, os.path.basename(file))
        except Exception as e:
            print(file)
            print(e)
        shutil.copyfile(file, output_path)

    exit_imgs_folder = os.path.join(folder_path, 'exit_images')
    os.makedirs(exit_imgs_folder, exist_ok=True)

    for file in exit_imgs_path:
        try:
            output_path = os.path.join(exit_imgs_folder, os.path.basename(file))
        except Exception as e:
            print(file)
            print(e)
        shutil.copyfile(file, output_path)

    entry_ids_file_path = os.path.join(folder_path, 'entry_ids.txt')
    with open(entry_ids_file_path, 'w') as entry_ids_file:
        for id in entry_ids_list:
            entry_ids_file.write(str(id) + '\n')

    exit_ids_file_path = os.path.join(folder_path, 'exit_ids.txt')
    with open(exit_ids_file_path, 'w') as exit_ids_file:
        for id in exit_ids_list:
            exit_ids_file.write(str(id) + '\n')

    top_10_ids_file_path = os.path.join(folder_path, 'top_10_ids.txt')
    with open(top_10_ids_file_path, 'w') as top_10_ids_file:
        for id in ids:
            top_10_ids_file.write(str(id) + '\n')

    df_file_path = os.path.join(folder_path, 'data.csv')
    df.to_csv(df_file_path, index=False)


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

def extractEntryImagesFeaturesDatabase(entry_ids_list):
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    mycol = mydb["Entry"]
    # for entry_id in entry_ids_list:
    find_query = {'engAssignedID': {'$in': entry_ids_list}}
    entry_infos = list(mycol.find(find_query))
    print(entry_infos)
    temp = []
    entry_filenames, entry_images_features, entry_images = [], [], []
    for entry_info in entry_infos:
        id, filenames = entry_info["engAssignedID"], entry_info["filename"]
        for file in filenames:
            entry_images.append(file)
            entry_filenames.append(id)
            features = calculate_target_feature(file)
            entry_images_features.append(features)
    myclient.close()
    # print(len(entry_images_features))
    return entry_filenames, entry_images_features, entry_images

def extractExitImagesFeaturesDatabase(exit_id):
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
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    mydb = myclient["CVAutonomo"]
    mycol = mydb["Exit"]
    find_query = {'cvAssignedID': {'$eq': exit_id}}
    exit_infos = list(mycol.find(find_query))
    exit_filenames = []
    exit_images_features = []
    exit_images = []
    for exit_info in exit_infos:
        print(exit_info)
        id, filenames = exit_info["cvAssignedID"], exit_info["filename"]
        for file in filenames:
            exit_images.append(file)
            exit_filenames.append(id)
            features = calculate_target_feature(file)
            exit_images_features.append(features)
    myclient.close()
    # print(exit_images_features)
    return exit_filenames, exit_images_features, exit_images

def RecommentationList(entry_ids_list, exit_id):
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
    entry_filenames, entry_images_features, entry_images = extractEntryImagesFeaturesDatabase(entry_ids_list)
    exit_filenames, exit_images_features, exit_images = extractExitImagesFeaturesDatabase(exit_id)
    print("Length of Entry and Exit Filenames: ",len(entry_filenames), len(exit_filenames))
    complete_list = []
    for i in range(0, len(exit_filenames)):
        for j in range(0, len(entry_filenames)):
            temp = []
            target_embedding = exit_images_features[i]
            entry_image = entry_images_features[j]
            matching_results =  cosine_distances(target_embedding, entry_image)
            # print("Matching Results: ", matching_results)
            temp.append(exit_filenames[i])
            temp.append(entry_filenames[j])
            matching_value = round((1-matching_results[0][0]), 2)
            matching_value = matching_value * 100
            temp.append(matching_value)
            complete_list.append(temp)
    df = pd.DataFrame(complete_list, columns=["exit_id", "entry_id", "conf"])
    # print(df.head())
    temp = df.sort_values(by=['conf'], ascending=False)
    top_10_ids = list(temp["entry_id"])[:10]
    top_10_conf = list(temp["conf"])[:10]
    top_10_conf = [round(x) for x in top_10_conf]
    print(top_10_conf, top_10_ids)
    persons_recommended = {}
    for id in range(len(top_10_ids)):
        if not top_10_ids[id] in persons_recommended.keys():
            persons_recommended[top_10_ids[id]] = top_10_conf[id]
    print("Top 10 Ids: ", top_10_ids)
    print("Persons: ", persons_recommended)
    return top_10_ids, persons_recommended, entry_filenames, exit_filenames, entry_images, exit_images, df

def exitTime_utc(exitTime):
    tm = datetime.datetime.strptime(str(exitTime),"%Y%m%dT%H%M%S.%f")
    tm_utc = tm.astimezone(pytz.UTC)
    tm_utc = tm_utc - timedelta(seconds=10)
    tm_utc = tm_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return tm_utc

def minChange(prev_min):
    time_now = datetime.datetime.now()
    current_time = time_now.strftime("%Y%m%d_%H%M%S.%f")
    current_min = current_time.split(".")[0][-4:-2]
    if prev_min == current_min:
        return False
    else:
        return True

def main():
    counter = 0
    base_path = "/home/orange/assets/atn-bako-001"
    time_prev = datetime.datetime.now()
    prev_time = time_prev.strftime("%Y%m%d_%H%M%S.%f")
    prev_min = prev_time.split(".")[0][-4:-2]
    while True:
        minChangeFlag = minChange(prev_min)
        # if minChangeFlag == False:
        #     continue
        temp_time_prev = datetime.datetime.now()
        temp_prev_time = temp_time_prev.strftime("%Y%m%d_%H%M%S.%f")
        prev_min = temp_prev_time.split(".")[0][-4:-2]
        day = date.today()
        day = str(day.strftime("%Y%m%d"))
        myquery = { "RecommendedShopperID":{'$size':0}}
        
        for doc in exit_features.find(myquery):
            exit_id = doc["cvAssignedID"]
            exit_time = doc["exitTimestamp"]
            entry_ids = doc["openEntriesID"]
            if len(entry_ids) == 0:
                continue
            exitimg_filepath = doc["filename"][0]
            exit_eng_id = doc["engAssignedID"]
            ids, persons_recommended, entry_ids_list, exit_ids_list, entry_imgs_path, exit_imgs_path, df = RecommentationList(entry_ids, exit_id)
            print("Recommended IDS: ", ids, persons_recommended)
            entry_imgs_count, exit_imgs_count = len(entry_ids_list), len(exit_ids_list)
            f(counter,exit_id,exit_time,entry_ids,exitimg_filepath,exit_eng_id,ids,entry_ids_list,exit_ids_list,entry_imgs_path,exit_imgs_path,df)
            counter+=1
            print("-"*100)
            try:
                tm = datetime.datetime.strptime(str(exit_time),"%Y-%m-%d %H:%M:%S")
            except Exception as e:
                tm = datetime.datetime.strptime(str(exit_time),"%Y-%m-%d %H:%M:%S.%f")
            tm_utc = tm.astimezone(pytz.UTC)
            tm_utc = tm_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            rel_failed_img = str(os.path.relpath(exitimg_filepath, base_path))
            rel_failed_img = "/"+str(rel_failed_img)
            if len(persons_recommended) > 0:
                ids_count = Counter(ids)
                ids_count = dict(ids_count)
                threshold_count = exit_imgs_count * 3
                if threshold_count > 9:
                    threshold_count = 9
                elif threshold_count < 0:
                    threshold_count = 6
                    
                temp = [k for k,v in ids_count.items() if int(v) >= threshold_count]                
                print(temp)

                rel_failed_img = str(os.path.relpath(exitimg_filepath, base_path))
                rel_failed_img = "/"+str(rel_failed_img)
                # exit()
                if len(temp) != 1 :
                    # print(ids_count)
                    max_value = max(list(ids_count.values()))
                    # max_value = int(max(ids_count, key=ids_count.get))
                    recommended_ids = [k for k,v in ids_count.items() if int(v) >= max_value]
                    entryMatching = {}
                    for recom_id in recommended_ids:
                        entryMatching[recom_id] = persons_recommended[recom_id]
                    myquery = { "cvAssignedID": exit_id }
                    processedTime = datetime.datetime.now()
                    newvalues = { "$push": {"processedTime": processedTime, "RecommendedShopperID": entryMatching}}
                    print("Exit Id and ENtry matching1: ",exit_id, entryMatching)
                    autonomoEntryExitActivityHandler.confirmUserExit(None, tm_utc, None, exit_eng_id, entryMatching)
                    exit_features.update_one(myquery, newvalues)
                    
                elif len(temp) == 1:
                    
                    """
                    Auto Exit Person Matched
                    """
                    entryMatching = {}
                    for final_id in temp:
                        entryMatching[final_id] = persons_recommended[final_id]
                    matching_id = temp[0]
                    conf = persons_recommended[matching_id]
                    entry_id_status = autonomoEntryExitActivityHandler.getUserActivity(matching_id)
                    print()
                    print("="*100)
                    print("Confidence Status: ", conf)
                    # print("Entry ID Status: ", entry_id_status)
                    """
                    TO Auto Exit the Shopper Uncomment the below lines
                    """
                    # if len(temp) == 1 and conf >= 79 and entry_id_status != "CHECKEDOUT" and entry_id_status != "PROCESSED":
                    #     # if conf >= 76 and entry_id_status != "CHECKEDOUT" and entry_id_status != "PROCESSED":
                    #     print("Auto Exited the User: ", matching_id)
                    #     print("Matching Count: ", ids_count)
                    #     print("*"*100)
                    #     print()
                    #     autonomoEntryExitActivityHandler.confirmUserExit(matching_id, tm_utc, None, exit_id, None)
                    #     myquery = { "id": matching_id }
                    #     newvalues = { "$set": {"AutoExited": True}}
                    #     entry_features.update_one(myquery, newvalues)

                    # else:
                    print("Couldnt auto Exit user")
                    print("Matching Count: ", ids_count)
                    print("*"*100)
                    autonomoEntryExitActivityHandler.confirmUserExit(None, tm_utc, None, exit_eng_id, entryMatching)
                    myquery = { "cvAssignedID": matching_id }
                    newvalues = { "$set": {"AutoExited": False}}
                    # entry_features.update_one(myquery, newvalues)
                    
                    myquery = { "cvAssignedID": exit_id }
                    processedTime = datetime.datetime.now()
                    newvalues = { "$push": {"RecommendedShopperID": entryMatching}}
                    print("Exit Id and ENtry matching2: ",exit_id, entryMatching)
                    exit_features.update_one(myquery, newvalues)
                    
                    
            else:
                entryMatching = {}
                myquery = { "cvAssignedID": exit_id }
                processedTime = datetime.datetime.now()
                newvalues = { "$push": {"processedTime": processedTime, "RecommendedShopperID": None}}
                print("Exit Id and ENtry matching3: ",exit_id, entryMatching)
                autonomoEntryExitActivityHandler.confirmUserExit(None, tm_utc, [rel_failed_img], exit_eng_id, entryMatching)
                exit_features.update_one(myquery, newvalues)
            print(entryMatching)
                

if __name__ == "__main__":
    main()