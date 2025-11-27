import requests
import configparser
from datetime import datetime, timedelta

config = configparser.RawConfigParser()
config.read('/home/orange/Production/entryExitMultipleCam/AS-One/application.properties')

base_url = config.get('ConfigurationSection', 'base_url')

def getShopperOpenCartEvents(userActivityId):

    try :
        api_url = base_url + "/api/store-carts/shopper-open-events?userActivityId.equals=" + userActivityId + "&sort=eventTime,desc&size=100";

        response = requests.get(api_url)
        responseJson = response.json()
        # print(responseJson)
        return responseJson
    except Exception as e:
        print(e)
        return None

def getOpenShoppers(eventTime):
    try:
        activeShoppersUrl = base_url + "/api/user-activities?eventStatus.in=AUTHORIZED,TRACKED&entryTime.lessThan=" + eventTime + "&sort=entryTime,desc"

        response = requests.get(activeShoppersUrl)
        responseJson = response.json()

        checkedoutShoppersUrl = base_url + "/api/user-activities?eventStatus.equals=CHECKEDOUT&exitTime.greaterThan=" + eventTime + "&sort=entryTime,desc"

        res = requests.get(checkedoutShoppersUrl)
        resJson = res.json()
        
        result = responseJson + resJson
        
        # print(responseJson)
        return result
    except Exception as e:
        print(e)
        return None

def assignRecommendedShoppers(recommendationInput):

    storeCartId = recommendationInput['eventID']
    activityDict = recommendationInput['activityList']

    recommendedShoppers = list()

    for key, value in activityDict.items():
        recommendedShopper = {}
        recommendedShopper['userActivityId'] = key
        recommendedShopper['confidenceScore'] = value
        recommendedShoppers.append(recommendedShopper)


    payload = {}
    payload['id'] = storeCartId
    
    payload['recommendedShoppers'] = recommendedShoppers

    try:
        api_url = base_url + "/api/store-carts/" + storeCartId + "/assign-recommended-shoppers"
        
        response = requests.put(api_url, json=payload)
        return response

    except Exception as e:
        print(e)
