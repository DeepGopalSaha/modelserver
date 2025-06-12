import requests
import json
import os
#distance must be in meters. Less than equal to 30km
#{"error":"Unable to geocode"} : no data available


api_key=str(os.environ.get("POI_API_KEY"))
api_key=r"pk.dda5c5c7f1ffd0a555609e3142351ff8"
def poi(curlat: float, curlng: float,id: int=-1, responsetype: str="%2A", dist :int=5000, limit: int=10):
    url_type={
        0: "%2A", 1:"airport", 2:"restaurant", 3:"bank", 4:"atm", 5:"hotel", 6:"pub", 7:"bus_station", 
        8:"railway_station", 9:"cinema", 10:"hospital", 11:"college", 12:"school", 13:"pharmacy", 14:"supermarket", 
        15:"fuel", 16:"gym", 17:"place_of_worship", 18:"toilet", 19:"park", 20:"stadium", 21:"parking", 22:"cardealer"
    }
    
    #request_type = url_type[id]
    if id ==-1:
        request_type=responsetype
    else:
        if id not in url_type:
            return 400,"{'given id is not in list'}"
        request_type = url_type[id]    
                
    url = "https://us1.locationiq.com/v1/nearby?lat={}&lon={}&tag=amenity%3A{}&radius={}&limit={}&key={}".format(curlat, curlng , request_type, dist,limit,api_key)
    headers = {"accept": "application/json"}
    #print(url)

    response = requests.get(url, headers=headers)
    poi_text = json.loads(response.text)
    print(poi_text)
    places_dict={}
    if response.status_code==200:
        for places in poi_text:
            temp_name='Not Available'
            tmep_display_name='Not Available'
            if 'name' in places.keys():
                temp_name=places['name']
            if 'display_name' in places.keys():
                tmep_display_name=places['display_name']    
            temp_dict={
                        'name': temp_name,
                        'display_name': tmep_display_name,
                        'lat_long_coordinate': [places['lat'],places['lon']],
                        'distance': places['distance']
                        }
            temp_type=places['type']
            if temp_type not in places_dict.keys():
                places_dict[temp_type]=[]
            places_dict[temp_type].append(temp_dict)    

    JsonPOIDetails =  json.dumps(places_dict)
    return response.status_code, JsonPOIDetails


if __name__=='__main__':
    print(poi(curlat=22.5837, curlng=88.34285, id=15, dist=1500))
    