# MDSTN (Multi-Directional Short-Term Traffic Volume Prediction based on Spatio-Temporal Networks)

### Note
* This repository is **no longer maintained**, as it was finished a long time ago (2019 ~ 2020). It is recommended to utilize the latest technologies for traffic prediction, such as GNNs and Transformers, instead of CNNs. Good luck!

### Prepare Data
* Step 1. Download the [NYCBike](https://data.cityofnewyork.us/) dataset.
* Step 2. Process the POI data by getPOI.py, POIFromCSV.py in the folder of preprocess.
* Step 3. Process the traffic volume by ToMapNYCBike.py and ProcessInOutFlow.py.
* Step 4. Modify the file path in MDSTN.py.

### Run MDSTN
    python MDSTN.py

### Dependencies
* Tensorflow > 2.0.0
* [Anaconda](https://www.anaconda.com/) 4.8.2 contains all the required packages.
