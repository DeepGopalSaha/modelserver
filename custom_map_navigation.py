import geopy.distance
import json
import requests
import time
import os

class map_navigation:
    def __init__(self):
        self.starting=None
        self.ending=None
        self.epsilon=0.000005
        self._initialise_class_vars()
        self.api_key=str(os.environ.get("CUSTOM_MAP_API_KEY"))        

    def _initialise_class_vars(self):
        self.map_text=None
        self.distance=0
        self.waypoints=[]
        self.route_dictionary={}  #format {lower checkpoint : {"distance":54.5,"duration":13.1,"type":11,"instruction":"Head south","name":"-","way_points":[12,16] }
        self.lastcheckpoint=0
        self.nextcheckpoint=0
        self.last_refactoring=time.time()

    def create_nav_map(self,starting: list , ending: list) :
        #generate the route text and store it in map_text
        self._initialise_class_vars()
        self.starting=starting
        self.ending=ending
        '''
        map_text={"type":"FeatureCollection",
                  "bbox":[88.343597,22.565375,88.370852,22.585672],
                  "features":[{"bbox":[88.343597,22.565375,88.370852,22.585672],
                               "type":"Feature",
                               "properties":{
                                   "segments":[{"distance":4531.6,
                                                "duration":527.8,
                                                "steps":[
                                                    {"distance":54.5,"duration":13.1,"type":11,"instruction":"Head south","name":"-","way_points":[0,1]},
                                                    {"distance":236.7,"duration":44.3,"type":0,"instruction":"Turn left","name":"-","way_points":[1,12]},
                                                    {"distance":136.7,"duration":23.3,"type":1,"instruction":"Turn right onto Parikshit Roy Lane","name":"Parikshit Roy Lane","way_points":[12,16]},
                                                    {"distance":341.6,"duration":35.1,"type":1,"instruction":"Turn right onto Vidyapati Flyover","name":"Vidyapati Flyover","way_points":[16,19]},
                                                    {"distance":2716.9,"duration":262.5,"type":0,"instruction":"Turn left onto Vidyapati Flyover","name":"Vidyapati Flyover","way_points":[19,66]},{"distance":822.7,"duration":112.7,"type":1,"instruction":"Turn right","name":"-","way_points":[66,78]},{"distance":38.7,"duration":9.3,"type":0,"instruction":"Turn left","name":"-","way_points":[78,80]},{"distance":93.6,"duration":11.2,"type":0,"instruction":"Turn left onto Lower Foreshore Road","name":"Lower Foreshore Road","way_points":[80,85]},{"distance":19.2,"duration":3.4,"type":13,"instruction":"Keep right","name":"-","way_points":[85,86]},{"distance":71.0,"duration":12.8,"type":13,"instruction":"Keep right","name":"-","way_points":[86,89]},
                                                    {"distance":0.0,"duration":0.0,"type":10,"instruction":"Arrive at your destination, on the right","name":"-","way_points":[89,89]}
                                                    ]}],
                                    "way_points":[0,89],
                                    "summary":{"distance":4531.6,"duration":527.8}\
                                        },
                                "geometry":{"coordinates":[[88.37046,22.567639],[88.370623,22.567172],[88.370668,22.567155],[88.3708,22.567039],[88.370852,22.566933],[88.370842,22.56688],[88.370817,22.566817],[88.370662,22.566691],[88.370531,22.566447],[88.370302,22.566269],[88.370176,22.565901],[88.370012,22.565416],[88.369994,22.565375],[88.369899,22.56541],[88.369552,22.565549],[88.368852,22.565822],[88.368765,22.565846],[88.368784,22.56589],[88.369472,22.567385],[88.370103,22.568659],[88.369307,22.569936],[88.368973,22.570576],[88.36888,22.570717],[88.368821,22.570851],[88.368649,22.571099],[88.368503,22.571361],[88.368286,22.571684],[88.368214,22.571804],[88.367967,22.572165],[88.367769,22.572485],[88.367194,22.573371],[88.366774,22.573991],[88.366669,22.574158],[88.366283,22.574787],[88.366175,22.57495],[88.365701,22.575598],[88.364994,22.576371],[88.364915,22.576444],[88.364316,22.576866],[88.364093,22.577036],[88.363245,22.577537],[88.362681,22.577901],[88.362273,22.578144],[88.361918,22.578301],[88.361844,22.578322],[88.361521,22.578411],[88.360746,22.578628],[88.360653,22.578656],[88.360531,22.57872],[88.359837,22.578903],[88.35951,22.57899],[88.358746,22.579199],[88.358208,22.579289],[88.357726,22.579435],[88.357444,22.579524],[88.35682,22.57972],[88.356526,22.579875],[88.355716,22.580239],[88.355322,22.580392],[88.35513,22.580462],[88.354482,22.580681],[88.354082,22.580915],[88.353761,22.581041],[88.352347,22.581493],[88.351964,22.581627],[88.351222,22.581858],[88.349788,22.582305],[88.349624,22.582662],[88.349615,22.583236],[88.349572,22.583659],[88.349529,22.583838],[88.349554,22.584001],[88.349556,22.584167],[88.349517,22.584304],[88.349294,22.584484],[88.344658,22.585621],[88.344272,22.585672],[88.344114,22.585645],[88.343987,22.58556],[88.343965,22.585423],[88.34402,22.58522],[88.344074,22.585211],[88.344127,22.585158],[88.344189,22.584774],[88.344184,22.584639],[88.344112,22.584455],[88.343991,22.584324],[88.343836,22.584205],[88.343715,22.584043],[88.343597,22.583814]],"type":"LineString"}}],"metadata":{"attribution":"openrouteservice.org | OpenStreetMap contributors","service":"routing","timestamp":1741616896665,"query":{"coordinates":[[88.371,22.5678],[88.3434,22.5839]],"profile":"driving-car","profileName":"driving-car","format":"geojson"},"engine":{"version":"9.0.0","build_date":"2025-01-27T14:56:02Z","graph_date":"2025-01-28T09:38:16Z"}}}
        '''
        body = {"coordinates":[starting[::-1],ending[::-1]]}
        headers = {'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
                    'Authorization': self.api_key,
                    'Content-Type': 'application/json; charset=utf-8'
                    }
        call = requests.post('https://api.openrouteservice.org/v2/directions/driving-car/geojson', json=body, headers=headers)
        map_text=json.loads(call.text)
        print(map_text)
        route=map_text["features"][0]["properties"]["segments"][0]["steps"]
        
        #self.waypoints=[[map_text['bbox'][0],map_text['bbox'][1]]]
        self.waypoints=[self.starting]
        self.waypoints.extend([i[::-1] for i in map_text["features"][0]["geometry"]["coordinates"]])
        #self.waypoints.append([map_text['bbox'][2],map_text['bbox'][3]])
        self.waypoints.append(self.ending)
        
        self.route_dictionary[0]={"distance":0,"duration":0,"type":-1,"instruction":"Go Forward","name":"-","way_points":[0,1]}
        for steps in route:
            self.route_dictionary[steps["way_points"][0]+1]=steps
        
        temp_dict={'waypoints': self.waypoints}
        temp_dict.update(map_text["features"][0]["properties"]["summary"])
        '''
        temp_dict -> {'waypoints': ...,
                       "distance":4531.6,
                       "duration":527.8 
                     }
        '''

        jsonWaypointsString=json.dumps(temp_dict)
        return jsonWaypointsString
    

    def _distance(self,a: list,b: list):
        [x1,y1]=a
        [x,y]=b
        return ((x1 - x)**2 + (y1 - y)**2)**0.5
    
    def _check_is_between(self,a: list,b: list,c: list):
        #epsilon=0.000001
        return abs(self._distance(a,b)+self._distance(b,c)-self._distance(a,c))<self.epsilon
        
    
    def check_nav_map(self,cur_gps: list):
        [x,y]=cur_gps
        #if self.lastcheckpoint>=len(self.waypoints)-1:
        #    self.create_nav_map(cur_gps,self.ending)
        #    return 
        '''
        if self.lastcheckpoint==-1:
            (x1,y1)=self.starting
        else:
            (x1,y1)=self.waypoints[self.lastcheckpoint]     
        '''

        [x1,y1]=self.waypoints[self.lastcheckpoint]    
        [x2,y2]=self.waypoints[self.lastcheckpoint+1]

        while not self._check_is_between((x1,y1),(x,y),(x2,y2)):
            self.lastcheckpoint+=1
            if self.lastcheckpoint>=len(self.waypoints)-2 :
                if time.time()-self.last_refactoring<120:
                    self.epsilon*=2
                return (self.create_nav_map(cur_gps,self.ending))    
                
                    
            [x1,y1]=self.waypoints[self.lastcheckpoint]    
            [x2,y2]=self.waypoints[self.lastcheckpoint+1]  
        #lat-long distance
        instruction=''
        for key in self.route_dictionary.keys():
            if self.lastcheckpoint <self.route_dictionary[key]['way_points'][1]:
                instruction=self.route_dictionary[key]['instruction']
                self.nextcheckpoint=self.route_dictionary[key]['way_points'][1]
                break

        self.distance=geopy.distance.geodesic(cur_gps,self.waypoints[self.nextcheckpoint]).meters
        jsonString=json.dumps([instruction,self.distance])
        return jsonString


        try:
            self.mapBuffer.put([self.route_dictionary[self.lastcheckpoint]['Type'],
                            self.route_dictionary[self.lastcheckpoint]['instruction'],
                            self.distance], block=True, timeout=0.5)
            return ([self.route_dictionary[self.lastcheckpoint]['Type'],
                            self.route_dictionary[self.lastcheckpoint]['instruction'],
                            self.distance])
        except Exception as e:
            print('Mapbuffer error')
            print(e)
            pass

#map=map_navigation()

map_pool={}
TTL_time=600 # the map will be deleted after an inactivity of 10min

def clear_map(id=None):
    global map_pool
    if id==None:
        map_pool={}
    else:        
        map_pool.pop(id, None)

def check_for_inactive_id():
    global map_pool
    to_be_deleted_id=[]
    temp_cur_time=time.time()
    for id in map_pool.keys():
        if map_pool[id]['TTL']<temp_cur_time:
            to_be_deleted_id.append(id)
    for i in to_be_deleted_id:
        clear_map(i)        
    print("DELETD IDS -> ",to_be_deleted_id)


def create_map(starting_latlong: list,ending_latlong: list, id = 0):
    global map_pool, TTL_time
    check_for_inactive_id()
    map_pool[id]={'obj':map_navigation(), "TTL":time.time()+TTL_time} 
    print("CURRENT MAP POOL -> ",map_pool)
    return(map_pool[id]['obj'].create_nav_map(starting=starting_latlong,ending=ending_latlong))

def check_map(cur_gps,id=0):
    global map_pool,TTL_time
    check_for_inactive_id()
    if id in map_pool.keys():
        map_pool[id]['TTL']+=TTL_time
        return (map_pool[id]['obj'].check_nav_map(cur_gps))
    else:
        return ('Map not found')

if __name__=='__main__':
    create_map([22.565375,88.343597],[22.585672, 88.370852],id=10)
    print(map_pool)
    a=[88.367718,22.574312][::-1]
    #print(check_map(a,id=10))
    time.sleep(6)
    check_for_inactive_id()
    print(map_pool)
    
    print(check_map(a,id=10))

        