import requests
import os
import subprocess
import json
from enum import Enum
import pyproj
from pyproj import Proj
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import date

from tqdm import tqdm

import pandas as pd
from concurrent.futures import ThreadPoolExecutor


base_url = 'https://data.chc.ucsb.edu/products/CHIRTSdaily/v1.0/global_tifs_p05/'
cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache'))

class CHIRTSvariable(Enum):
#    HeatIndex = 'HeatIndex'
#    RHum = 'RHum'
    Tmax = 'Tmax'
    Tmin = 'Tmin'
    svp = 'svp'
    vpd = 'vpd'



def _acquire_data(variable: CHIRTSvariable, longitude: float, latitude: float, year: int, month: int, day: int, timeout=10  ) -> float:
    cog_url = f'{base_url}/{variable.value}/{year}/{variable.value}.{year:04d}.{month:02d}.{day:02d}.tif'
    # use gdallocationinfo to extract the value at the pixel
    cmd = ['gdallocationinfo', '-valonly', '/vsicurl/' + cog_url, '-wgs84', str(longitude), str(latitude)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode != 0:
        raise ValueError(f'Error acquiring data:  {cog_url} {result.stderr}')
    
    return float(result.stdout)


class CHIRTSdaily:
    def __init__(self, longitude, latitude, datadir='./'):
        self.datadir = datadir
        self.longitude = longitude
        self.latitude = latitude
        self.df = None
        (self._px, self._py), (self._longitude, self._latitude) = self.get_nearest_pixel_coords()

    def acquire_data(self, force_fetch=False) -> float:
        # check cache
        cache_file = os.path.join(cache_dir, f'{self._px}_{self._py}.parquet')
        if os.path.exists(cache_file) and not force_fetch:
            self.df = pd.read_parquet(cache_file)
        else:
            self.df = self._acquire_timeseries()
            self.df.to_parquet(cache_file)

        return self.df

    def _acquire_timeseries(self, num_threads=60, progress_bar=True) -> pd.DataFrame:
        variables = [v for v in CHIRTSvariable]
        available_years = self.get_available_years(variables[0])
        startdate = date(available_years[0], 1, 1)
        enddate = date(available_years[-1], 12, 31)

        def fetch_data(date, variable):
            try:
                value = _acquire_data(variable, self.longitude, self.latitude, date.year, date.month, date.day)
                return {"date": date, "variable": variable.value, "value": value}
            except ValueError as e:
                print(f'Error acquiring data for {date}: {e}')
                return {"date": date, "variable": variable.value, "value": None}

        dates = pd.date_range(startdate, enddate)
        tasks = [(date, variable) for date in dates for variable in variables]
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(tqdm(executor.map(lambda task: fetch_data(*task), tasks), total=len(tasks), disable=not progress_bar))

        # Directly create a DataFrame from the list of dictionaries
        df = pd.DataFrame(results)

        # Pivot the DataFrame to have dates as index, variables as columns
        df_pivot = df.pivot(index='date', columns='variable', values='value')

        return df_pivot

    def get_nearest_pixel_coords(self):
        
        template_cog = f'{base_url}/Tmax/2016/Tmax.2016.01.01.tif'
         
        cmd = ['gdalinfo', '-json', '/vsicurl/' + template_cog]
        result = subprocess.run(cmd, stdout=subprocess.PIPE)
        data = json.loads(result.stdout)
        transform = data['geoTransform']
        width, height = data['size']

        x = self.longitude
        y = self.latitude

        # Apply the affine transformation to get pixel coordinates
        # Note: Affine transform is in the form of (a, b, c, d, e, f)
        # where x' = a * x + b * y + c and y' = d * x + e * y + f
        a, b, c, d, e, f = transform
        col = (x - a) / b
        row = (y - d) / f

        # Get nearest pixel center
        col, row = round(col), round(row)

        # Check if the pixel is within the bounds of the raster
        if 0 <= col < width and 0 <= row < height:
            # Convert the pixel position back to geographic coordinates
            return (col, row), (col * b + a, row * f + d)
        else:
            return (None, None), (None, None)

    def get_available_years(self, variable: CHIRTSvariable):
        response = requests.get(base_url + variable.value)
        response.raise_for_status()
    
        soup = BeautifulSoup(response.text, "html.parser")
        
        links = [
            urljoin(base_url, a['href'])
            for a in soup.find_all('a', href=True)
            if not a['href'].startswith('/..') and 'Parent Directory' not in a.text
        ]
        
        # Extract the year from the links
        years = []
        for link in links:
            if link.endswith('/'):
                years.append(int(link.split('/')[-2]))

        return sorted(years)

    def calculate_monthly_normals(self):
        data = self.df
        return data.groupby(data.index.month).mean()


if __name__ == '__main__':
    from time import time

    # Example usage
    chirts = CHIRTSdaily(-117.5, 46.5)

    t0 = time()
    print(chirts.acquire_data())

    elapsed = time() - t0
    print(f'Time elapsed: {elapsed:.2f} seconds')

    print(chirts.calculate_monthly_normals())