import os

import rasterio
from uszipcode import SearchEngine


def main(folder):
    with open('files_metadata.csv', mode='w') as f:
        f.write('file name,lat,lon,zip code')
        for file in [x for x in os.listdir(folder) if x.endswith('.tif')]:
            with rasterio.open(file) as src:
                lon_diff_half = (src.bounds.left - src.bounds.right) / 2
                lat_diff_half = (src.bounds.top - src.bounds.bottom) / 2
                lon = src.bounds.left - lon_diff_half
                lat = src.bounds.top - lat_diff_half
                search = SearchEngine(simple_zipcode=True)
                try:
                    zipcode = search.by_coordinates(lat,lon, radius=20, returns=1)[0].zipcode
                except IndexError:
                    zipcode = 'No zip code found'
                f.write(f'{os.path.basename(file)},{lat},{lon},{zipcode}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('folder', help='The path to the folder')

    main(parser.parse_args().folder)
