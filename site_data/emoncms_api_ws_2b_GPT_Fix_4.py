from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic('reset', '-sf')

import json
import requests
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from tqdm import tqdm


def get_emoncms_data(FEED_ID, FEED_TYP, DURATION, INTERVAL, start_time, apikey):
    # Convert start_time to Unix timestamp in milliseconds
    UNIX_MILLI_START = int(datetime.timestamp(start_time) * 1000)
    UNIX_MILLI_END = int(UNIX_MILLI_START + (DURATION * 60000))

    # Create payload for API request
    payload = {'id': FEED_ID, 'apikey': apikey, 'start': UNIX_MILLI_START, 'end': UNIX_MILLI_END, 'interval': INTERVAL}
    header = {'content-type': 'application/json'}
    
    # Make API request
    response = requests.get("http://emoncms.org/feed/data.json", params=payload, headers=header)

    # Handle errors
    if response.status_code != 200:
        print(f"Error fetching data for feed {FEED_ID} at {start_time}: {response.status_code}")
        return np.zeros((int(60 / DATAINT), 1))

    try:
        mylist = json.loads(response.content.decode("utf-8"))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response for feed {FEED_ID} at {start_time}: {e}")
        return np.zeros((int(60 / DATAINT), 1))

    if not mylist:
        print(f"No data returned for feed {FEED_ID} at {start_time}")
        return np.zeros((int(60 / DATAINT), 1))

    # Initialize array for raw data
    tts = int((DURATION * 60) / INTERVAL + 1)
    TRel = np.full(tts, -99.0)

    # Populate TRel with valid readings
    for COUNT, data_point in enumerate(mylist):
        if COUNT < tts and data_point[0] == UNIX_MILLI_START + COUNT * INTERVAL * 1000:
            TRel[COUNT] = data_point[1] if data_point[1] is not None else -99.0

    # Interpolate missing values
    mask = TRel == -99
    if np.any(~mask):
        TRel[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), TRel[~mask])

    # Initialize array for results
    DRes = np.zeros((int(60 / DATAINT), 1))

    # Calculate results for each interval
    for td in range(int(60 / DATAINT)):
        index = int(td * DATAINT * 6) % len(TRel)
        if FEED_TYP == 'T':
            DRes[td, 0] = TRel[index]
        elif FEED_TYP == 'W':
            DRes[td, 0] = max(0., TRel[index + int(DATAINT * 6) - 1] - TRel[index])
        elif FEED_TYP in ['H', 'P']:
            DRes[td, 0] = max(0., np.mean(TRel[index:index + int(DATAINT * 6)]) / ((60. / DATAINT) * 1000.))  # kWh from W 
        elif FEED_TYP == 'M':
            DRes[td, 0] = max(0., np.amax(TRel[index:index + int(DATAINT * 6)]) / 1000.)  # kW
        elif FEED_TYP in ['KW']:
            DRes[td, 0] = max(0., np.mean(TRel[index:index + int(DATAINT * 6)]) / ((60. / DATAINT))) # kWh form kW 
        elif FEED_TYP == 'R':
                DRes[td, 0] = max(0., np.amax(TRel[index:index + int(DATAINT * 6)]))  # Units as Feed
        elif FEED_TYP == 'A': 
            DRes[td, 0] = np.mean(TRel[index:index + int(DATAINT * 6)])  # Calculate average value
     

    return DRes

#Call input Data 
EMONPI_APIKEY = '65b4a6080ccd5b41278cf34f06746c63'
INTERVAL = 15            #seconds
DURATION = 30            #seconds
DATAINT = 30             #seconds
IDS = [473138,473147]
IDT = ['A','A']
S_year=2023
S_month = 1             #from 1-12
S_day = 1               #ensure within month
NoDays = 7              #days

# Number of days per month, considering leap year for 2023
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

ORes = None

for go in range(len(IDS)):
    FEED_ID = IDS[go]
    FEED_TYP = IDT[go]
    TRes = []

    year = S_year
    month = S_month
    day = S_day 
    

    # Collect NoDays of data 
    
    for d in tqdm(range(NoDays), desc=f"Processing Feed {FEED_ID}"):
        for hr in range(24):
            START = datetime(year, month, day, hr, 0, 0)
            Res = get_emoncms_data(FEED_ID, FEED_TYP, DURATION, INTERVAL, START, EMONPI_APIKEY)

            if Res is not None and len(Res) > 0:
                if len(TRes) > 0:
                    TRes = np.vstack((TRes, Res))
                else:
                    TRes = Res
            else:
                print(f"No data returned for {START}")

        day += 1
        if day > days_per_month[month - 1]:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1

    if go == 0:
        ORes = TRes
    else:
        ORes = np.hstack((ORes, TRes))

start_date = datetime(S_year, S_month, S_day)
time_column = [(start_date + timedelta(minutes=i * DATAINT)).strftime('%d/%m/%Y %H:%M') for i in range(len(ORes))]

# Create the column titles based on IDS and type
data_columns = [
    f'Data_{IDS[i]}_{IDT[i]}_{"kW" if IDT[i] == "M" else "kW-as-feed" if IDT[i] == "R" else "kWh-from-kW" if IDT[i] == "KW" else "Unit-not-set-by-script"}' 
    for i in range(len(IDS))
    ]

df = pd.DataFrame({'Time': time_column, **{col: ORes[:, i] for i, col in enumerate(data_columns)}})
df.to_excel('FH_CB001-Cb002_Power-half-houravrage_2023.xlsx', index=False)
