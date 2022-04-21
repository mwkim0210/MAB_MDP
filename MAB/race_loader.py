import os
import json
import requests
import time

from config import *


def get_all_seasons():
    """
    Get summary of all available seasons with API
    """
    if os.path.exists('data/seasons_summary.json'):
        return
    else:
        request = requests.get("https://api.sportradar.us/formula1/trial/v2/en/seasons.json?api_key="+API_KEY)
        data = request.json()
        print('Got seasons summary, sleep for 1 second')
        time.sleep(1)
        with open('data/seasons_summary.json', 'w') as f:
            json.dump(data, f, indent=4)


def get_stage_summary(stage_id: str):
    """
    Get summary of a stage and save as .json file with API
    :param stage_id: stage ID
    """
    # Get summary of stage
    if os.path.exists('data/summary_'+stage_id+'.json'):
        return
    else:
        request = requests.get("https://api.sportradar.us/formula1/trial/v2/en/sport_events/sr:stage:"
                               +stage_id+"/summary.json?api_key="+API_KEY)
        data = request.json()
        print('Got summary '+stage_id+', sleep for 1 second')
        time.sleep(1)
        with open('data/summary_'+stage_id+'.json', 'w') as f:
            json.dump(data, f, indent=4)


def seasons2dict():
    """
    Get all season's name and ID from .json file
    :return: dictionary with key: season name, value: season ID
    """
    seasons_dict = dict()

    try:
        with open('data/seasons_summary.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        get_all_seasons()
        with open('data/seasons_summary.json', 'r') as f:
            data = json.load(f)

    for season in data['stages']:
        """
        season_name = season['description'] + " " + season['scheduled'][:4] if len(season['description']) < 10 \
            else season['description']  # fix season names w/o season year
        """
        season_year = season['scheduled'][:4]
        season_id = ""
        for c in season['id']:
            if c.isdigit():
                season_id = season_id + c
        seasons_dict[season_year] = season_id

    return seasons_dict


def season2dict(season_id: str):
    """
    Get one season's summary from .json file
    :param season_id: ID of season
    """
    season_dict = dict()

    with open('data/summary_'+season_id+'.json', 'r') as f:
        data = json.load(f)

    for stage in data['stage']['stages']:
        gp = stage['description']
        gp_id = ""
        for c in stage['id']:
            if c.isdigit():
                gp_id = gp_id + c
        season_dict[gp] = gp_id
    return season_dict


def get_all_sessions(season_dict: dict):
    """
    Get all session(FP1, FP2, FP3, Qualifying, Race) from API
    """
    session_list = ['FP1', 'FP2', 'FP3', 'Q', 'Race']
    session_dict = dict()  # dict of all gp sessions
    for key, value in season_dict.items():  # iterate all GPs
        with open('data/summary_'+value+'.json', 'r') as f:
            data = json.load(f)
        session_info = dict()
        for i, session in enumerate(data['stage']['stages']):
            session_id = ""
            for c in session['id']:
                if c.isdigit():
                    session_id = session_id + c
            try:
                with open('data/summary_'+session_id+'.json', 'r') as f:
                    session_data = json.load(f)
            except FileNotFoundError:
                get_stage_summary(session_id)
                with open('data/summary_'+session_id+'.json', 'r') as f:
                    session_data = json.load(f)
            session_info[session_list[i]] = session_data['stage']['competitors']

        session_dict[key] = session_info
    return session_dict


def load_race_data():
    try:
        with open('session_info.json', 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError:
        print('File not found, run race_loader.py to download file')


def load_drivers():
    try:
        driver_dict = dict()
        with open('data/summary_121558.json', 'r') as f:
            data = json.load(f)
        for i in range(16):
            driver_dict[str(i)] = data['stage']['competitors'][i]
        return driver_dict
    except FileNotFoundError:
        get_stage_summary('121558')


def main():
    os.makedirs('data', exist_ok=True)
    get_all_seasons()  # Get all seasons' info available at API
    seasons_dict = seasons2dict()  # change .json from the line above to dict
    id_2010 = seasons_dict['2010']  # id_2010: id value of 2010 season
    get_stage_summary(id_2010)  # get 2010 season info from API
    season_dict_2010 = season2dict(id_2010)  # change .json from the line above to dict
    # Get all race summaries from API
    for key, value in season_dict_2010.items():
        get_stage_summary(value)

    session_dict = get_all_sessions(season_dict_2010)
    with open('session_info.json', 'w') as f:
        json.dump(session_dict, f, indent=4)


if __name__ == "__main__":
    main()
