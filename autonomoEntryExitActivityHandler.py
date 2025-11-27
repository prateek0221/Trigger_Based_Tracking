import requests
import json
import configparser
from datetime import datetime, timedelta

config = configparser.RawConfigParser()
config.read('/home/orange/Production/entryExitMultipleCam/AS-One/application.properties')

base_url = config.get('ConfigurationSection', 'base_url')

def validateId(activityId):
    if activityId is None:
        return False
    
    if len(activityId) < 20:
        return False
    
    return True

def getUserActivitiesInReview(entryTime, timeLimitInhrs = 2):
    # timeLimitInhrs = 2 if (timeLimitInhrs is None) else timeLimitInhrs
    timeLimit = datetime.strptime(entryTime, "%Y-%m-%dT%H:%M:%SZ") - timedelta(hours=timeLimitInhrs)
    expiryTime = timeLimit.strftime("%Y-%m-%dT%H:%M:%SZ")
    api_url = base_url + "/api/user-activities?sort=entryTime,asc&eventStatus.in=AUTHORIZED&entryTime.lessThanOrEqual=" + entryTime + "&entryTime.greaterThanOrEqual=" + expiryTime;

    print(api_url)
    
    response = requests.get(api_url)
    print(response._content)
    responseJson = response.json()
    print(responseJson)
    return responseJson

def getRecentAuthorizedUser(trackedTime):
    api_url = base_url + "/api/user-activities?sort=entryTime,desc&entryTime.lessThanOrEqual=" + trackedTime + "&eventStatus.equals=AUTHORIZED"
    
    response = requests.get(api_url)
    responseJson = response.json()
    return responseJson[0] if len(responseJson) > 0 else None

def notifyUserActivityTracked(userActivityId, userImgPath):
    isValidId = validateId(userActivityId)

    if isValidId == False:
        return None

    getUserActivityUrl = base_url + "/api/user-activities/" + userActivityId

    response = requests.get(getUserActivityUrl)
    userActivity = response.json()

    userActivity['userImagePaths'] = userImgPath

    notifyTrackedUrl = base_url + "/api/user-activities/" + userActivityId + "/notify-tracked"
    notifyTrackedResponse = requests.put(notifyTrackedUrl, json=userActivity)
    return notifyTrackedResponse.json()

def assignImageToUserActivity(userActivityId, userImgPath, entryTime, refId, tag):
    isValidId = validateId(userActivityId)

    if isValidId == False:
        createCameraActivity(userImgPath, entryTime, refId, tag)
        return None

    api_url = base_url + "/api/user-activities/" + userActivityId + "/attach-image"

    payload = {}
    payload['imagePath'] = userImgPath
    payload['tag'] = tag

    # print(payload)
    response = requests.put(api_url, json=payload)
    # print(response.json())
    return response.json()

def getUserActivity(userActivityId):
    isValidId = validateId(userActivityId)

    if isValidId == False:
        return None

    api_url = base_url + "/api/user-activities/" + userActivityId

    response = requests.get(api_url)
    if response != None:
        responseJson = response.json()
        return responseJson
    else:
        return None

def getUserExitActivitiesInReview(exitTime):
    api_url = base_url + "/api/user-exit-activities?status=OPEN&sort=exitTime,desc&exitTime.greaterThanOrEqual=" + exitTime

    response = requests.get(api_url)
    responseJson = response.json()
    return responseJson

def confirmUserExit(userActivityId, exitTime, userImgPath, externalTransactionRef, userActivityDict):
    api_url = base_url + "/api/user-exit-activities"

    get_url = api_url + "?externalTransactionRef.equals=" + externalTransactionRef

    fetchResponse = requests.get(get_url)
    userExitActivityList = fetchResponse.json()

    if len(userExitActivityList) > 0:
        userExitActivity = userExitActivityList[0]

        if userImgPath != None and len(userImgPath) > 0:
            responseJson = assignImageToExit(userExitActivity['id'], userImgPath[0])
        elif userActivityDict != None and len(userActivityDict) > 0:
            responseJson = attachRecommendedShoppers(userExitActivity['id'], userActivityDict)
        elif userActivityId != None:
            responseJson = assignUserActivity(userExitActivity['id'], userActivityId)
    else:
        payload = {}
        payload['exitTime'] = exitTime
        payload['userImagePaths'] = userImgPath
        payload['status'] = 'OPEN'
        payload['userActivityId'] = userActivityId
        payload['externalTransactionRef'] = externalTransactionRef

        suggestedUserActivities = []

        if userActivityDict != None and len(userActivityDict) > 0:
            for (user, score) in userActivityDict.items():
                suggestedUserActivity = {
                    'userActivityId': user,
                    'confidenceScore': score
                }
                suggestedUserActivities.append(suggestedUserActivity)
        
        payload['suggestedUserActivities'] = suggestedUserActivities

        response = requests.post(api_url, json=payload)
        responseJson = response.json()
    # print(responseJson)
    return responseJson

def assignImageToUserExitActivity(userActivityId, userImgPath):
    isValidId = validateId(userActivityId)

    if isValidId == False:
        return None

    userExitActivities_api_url = base_url + "/api/user-exit-activities?userActivityId.equals=" + userActivityId

    userExitActivitiesResponse = requests.get(userExitActivities_api_url)
    userExitActivitiesList = userExitActivitiesResponse.json()

    userExitActivity = {}
    userExitActivityId = ''
    if (len(userExitActivitiesList) == 1):
        userExitActivity = userExitActivitiesList[0]
        userExitActivityId = userExitActivity['id']

        response = assignImageToExit(userExitActivityId, userImgPath)
        return response.json()

def assignImageToExit(userExitActivityId, userImgPath):
    api_url = base_url + "/api/user-exit-activities/" + userExitActivityId + "/attach-image"
    payload = {}
    payload['imagePath'] = userImgPath

    response = requests.put(api_url, json=payload)
    return response

def getUserExitActivityBasedOnUserActvity(userActivityId):
    isValidId = validateId(userActivityId)

    if isValidId == False:
        return None

    api_url = base_url + "/api/user-exit-activities?userActivityId.equals=" + userActivityId


    response = requests.get(api_url)
    if response != None:
        responseJson = response.json()
        return responseJson[0] if len(responseJson) > 0 else None
    else:
        return None

def attachRecommendedShoppers(userExitActivityId, userActivityDict):
    api_url = base_url + "/api/user-exit-activities/" + userExitActivityId + "/attach-recommended-shoppers"

    payload = {}
    payload['id'] = userExitActivityId

    suggestedUserActivities = []

    if userActivityDict != None and len(userActivityDict) > 0:
        for (user, score) in userActivityDict.items():
            suggestedUserActivity = {
                'userActivityId': user,
                'confidenceScore': score
            }
            suggestedUserActivities.append(suggestedUserActivity)
    
    payload['suggestedUserActivities'] = suggestedUserActivities

    response = requests.put(api_url, json=payload)
    return response

def assignUserActivity(userExitActivityId, userActivityId):
    api_url = base_url + "/api/user-exit-activities/" + userExitActivityId + "/assign-user-activity"

    payload = {}
    payload['id'] = userExitActivityId
    payload['userActivityId'] = userActivityId

    response = requests.put(api_url, json=payload)
    return response

def createCameraActivity(userImagePaths, entryTime, refId, tag):
    api_url = base_url + "/api/camera-activities"

    payload = {}
    payload['userImagePaths'] = userImagePaths
    payload['recommendedEntryTime'] = entryTime
    payload['externalRef'] = refId
    payload['type'] = 'USER_ENTRY_ACTIVTIY'
    payload['tag'] = tag

    response = requests.post(api_url, json=payload)
    return response.json()