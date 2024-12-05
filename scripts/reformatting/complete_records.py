"""
Reads in pre-formatted daily files.
Writes to a TSV formatted aggregate file with all measurements per station.
"""

import os

data_origin = "/Users/rzieber/Documents/3D-PAWS/Turkiye/debug/Konya_Monthly/"
#data_origin = r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye_Comparison\\data\\TSMS\\"
data_destination = "/Users/rzieber/Documents/3D-PAWS/Turkiye/debug/Konya_Complete/"
#data_destination = r"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye_Comparison\\output\\monthly\\"
station_directories = [ # Konya, Ankara
    #"station_TSMS00/", "station_TSMS01/", "station_TSMS02/"      # Adana
    "station_TSMS03/", "station_TSMS04/", "station_TSMS05/"      # Konya
]
station_files = [] # list of lists of filenames under each station directory [[TSMS06 files], [TSMS07 files], [TSMS08 files]]
for station in station_directories:
    # sorted by filename
    station_files.append(sorted([f for f in os.listdir(data_origin + station) if os.path.isfile(os.path.join(data_origin + station, f))]))




    
for i in range(len(station_files)): # go through each station
    print(f"Reading station {station_directories[i][:len(station_directories[i])-1]}")

    filename = station_directories[i][8:14]+'_Complete_Record.dat'
    with open(data_destination+station_directories[i]+filename, 'w', newline='') as outfile:
        outfile.write("year  mon  day  hour  min  int  bmp2_temp  bmp2_pres  bmp2_slp  bmp2_alt  bme2_hum  htu_temp  htu_hum  sth_temp  sth_hum  mcp9808  tipping  vis_light  ir_light  uv_light  wind_dir  wind_speed\n")

        for file in station_files[i]: # go through each daily file at each station
            with open(data_origin+station_directories[i]+file, 'r', encoding='utf-8', errors='replace') as infile:
                next(infile) # skip the header row
                for row in infile:
                    outfile.write(row)
    # open a file for writing
    # write the header row to it
    # open every single daily file from origin
    # skip the first (header) row
    # write all the rows to the outfile
