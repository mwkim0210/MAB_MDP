#!/usr/bin/env python
# coding: utf-8

# In[223]:


#!/usr/bin/env python
# coding: utf-8
# Auther: Hyeonsu Lyu
# Contact: hslyu4@gmail.com

# ## Fetch XML data from data.go.kr

import requests
import json
import xmltodict


# In[265]:


def download_race_horses(service_key: str, filters: list = ['hrNo', 'meet', 'noracefl', 'rcRank', 'rcDate',
                                                                'rcDist', 'rcNo', 'rcOrd', 'rcP1Odd', 'rcP2Odd', 'rcP3Odd', 'rcPlansu']):
    race_horse_url = 'http://apis.data.go.kr/B551015/API186/SeoulRace'

    page = 1
    
    race_horses = []
    race_horses = load_race_horses()
    while True:
        print(f'Downloading page {page}')

        race_horse_params ={'serviceKey' : service_key, 'pageNo' : str(page), 'numOfRows' : '5000', 'rc_date_fr' : '20160101'}

        response = requests.get(race_horse_url, params=race_horse_params)
        if response.status_code != 200:
            raise ConnectionError(f'status code not 200. Failed with code {response.status_code}')

        response_xml = response.content.decode('utf8')
        response_dict = xmltodict.parse(response_xml)
        try:
            if 'items' not in response_dict['response']['body'].keys() or response_dict['response']['body']['items'] is None:
                # If it exceeds the last page, horse_ability will have no items.
                break

            # race_horse (dict) race_horse_list (list)
            race_horse_list = response_dict['response']['body']['items']['item']
            for race_horse in race_horse_list:
                race_horse_filtered = {key: race_horse[key] for key in race_horse.keys() if key in filters }
                race_horses.append(race_horse_filtered)
        except KeyError:
            print(response_dict)
            
        # save horse abilities
        with open('race_horses.json', 'w') as f:
            json.dump(race_horses, f, ensure_ascii=False, indent=4)


        page += 1
    print(f'{len(race_horses) = }')

    # save horse abilities
    with open('race_horses.json', 'w') as f:
        json.dump(race_horses, f, ensure_ascii=False, indent=4)

    return race_horses


def download_horse_abilities(service_key : str, filters: list = ['hrNm', 'hrEngNm', 'rcCnt', 'fstCnt', 'sndCnt', 'trdCnt', 'forthCnt', 
                                                               'fifthCnt', 'outCnt', 'amt', 'condAmt', 'winRate', 'quinRate', 'avgWinDist']):
    horse_ability_url = 'http://apis.data.go.kr/B551015/API42/totalHorseInfo'

    page = 1

    horse_abilities = {}
    while True:
        print(f'Downloading page {page}')

        horse_ability_params = {'serviceKey' : service_key, 'pageNo' : str(page), 'numOfRows' : '5000' }

        response = requests.get(horse_ability_url, params=horse_ability_params)
        if response.status_code != 200:
            raise ConnectionError(f'status code not 200. Failed with code {response.status_code}')

        response_xml = response.content.decode('utf8')
        response_dict = xmltodict.parse(response_xml)

        if 'items' not in response_dict['response']['body'].keys() or response_dict['response']['body']['items'] is None:
            # If it exceeds the last page, horse_ability will have no items.
            break
            
        # horse_ability (dict) horse_ability_list (list)
        horse_ability_list = response_dict['response']['body']['items']['item']
        for horse_ability in horse_ability_list:
            horse_ability_filtered = {key: horse_ability[key] for key in horse_ability.keys() if key in filters }
            horse_abilities[horse_ability['hrNo']] = horse_ability_filtered

        page += 1
        
    print(f'{len(horse_abilities) = }')
            
    # save horse abilities
    with open('horse_abilities.json', 'w') as f:
        json.dump(horse_abilities, f, ensure_ascii=False, indent=4)

    return horse_abilities


# In[244]:


def load_horse_abilities(path: str = 'horse_abilities.json'):
    # load horse_abilities
    with open(path, 'r') as f:
        horse_abilities = json.load(f)
    return horse_abilities

def load_race_horses(path: str = 'race_horses.json'):
    with open(path, 'r') as f:
        race_horse = json.load(f)
    return race_horse


# In[245]:


def print_horse(horse: dict, filter_tags: list = None):
    """
    params:
        horse (dict): 
        filter_tags (str or list): tag(s) to print
    return (None)
    """
    # convert str to list
    if isinstance(filter_tags, str):
        filter_tags = [filter_tags]
    for element, value in horse.items():
        if filter_tags is not None:
            if element in filter_tags:
                print(f'{element:10s} {value}')
        else:
            print(f'{element:10s} {value}')


# In[290]:


def wrap_horses_per_race(horses: list, horse_abilities: dict):
    """
    params:
        horse (dict): Race data of the horses
    return races (dict)
    """

    races = {}
    for horse in horses:
        # Horses in rank 06 don't have ratings.
      #  if horse['rcRank'] == '06':
       #     continue
        # Get horse ability from horse_abilities
        horse_ability_dict = horse_abilities[horse['hrNo']]
        
        horse.update(horse_ability_dict)

        winrate = (int(horse['fstCnt'])+int(horse['sndCnt'])+int(horse['trdCnt']))/int(horse['rcCnt'])                     if int(horse['rcCnt']) != 0 else 0
        horse['winrate'] = winrate

        racecode = f"{horse['rcDate']}_{horse['rcNo']}"
        # race code = date_no
        if racecode in races.keys():
            races[racecode].append(horse)
        else:
            races[racecode] = [horse]

    imperfect_races_code = []
    for racecode, race in races.items():
        if len(race) != int(race[0]['rcPlansu']):
            print(f"{len(race)=}, {int(race[0]['rcPlansu']) =}")
            imperfect_races_code.append(racecode)
            continue
        races[racecode] = sorted(race, key = lambda horse: horse['winrate'], reverse=True)
        #print(f'{len(races[racecode]) = }')
    for racecode in imperfect_races_code:
        del(races[racecode])

    return races


# In[304]:


def load_races(path : str = 'races.json'):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            races = json.load(f)
    except FileNotFoundError:
        with open('key.ini', 'r') as f:
            service_key = f.read().splitlines()

        try:
            horse_abilities = load_horse_abilities()
        except FileNotFoundError:
            print('No horse_abilities.json. Downloading..')
            horse_abilities = download_horse_abilities(service_key)

        try:
            race_horses = load_race_horses()
        except FileNotFoundError:
            print('No race_horses.json. Downloading..')
            race_horses = download_race_horses(service_key)

        races = wrap_horses_per_race(race_horses, horse_abilities)
        with open('races.json','w') as f:
            json.dump(races, f, ensure_ascii=False, indent=4)
    return races
    


# In[299]:


def main():
    races = load_races()
    print(len(races))
    print(races)
    
    for racecode, horses in races.items():
        print(racecode)
        for horse in horses:
            print_horse(horse)
            print('\n')
        break


# In[300]:


def download_race(service_key = str):
    race_url = 'http://apis.data.go.kr/B551015/API186/SeoulRace'
    race_params ={'serviceKey' : service_key, 'pageNo' : '1', 'numOfRows' : '200', 'rc_date_fr' : '20160101'}
    response = requests.get(race_url, params=race_params)
    race_xml = response.content.decode('utf8')

    race = xmltodict.parse(race_xml)
    horses = race['response']['body']['items']['item']
    
    return horses


# In[302]:


if __name__ == '__main__':
    main()


# In[ ]:




