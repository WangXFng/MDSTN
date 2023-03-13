# MDSTN (Multi-Directional Short-Term Traffic Volume Prediction based on Spatio-Temporal Networks)


### Dependencies
* Tensorflow > 2.0.0
* [Anaconda](https://www.anaconda.com/) 4.8.2 contains all the required packages.

### Instructions

### 1. Preprocessing data
* Step 1. Download the [NYCBike](https://data.cityofnewyork.us/) dataset.
* Step 2. Process the POI data by getPOI.py, POIFromCSV.py in the folder of "preprocess"
* Step 3. Process the traffic volume by ToMapNYCBike.py and ProcessInOutFlow.py in the folder of "preprocess"
* Step 4. Modify the file path in MDSTN.py and run
> python MDSTN.py


### Note
* If there is any problem, please contact to kaysen@hdu.edu.cn.
