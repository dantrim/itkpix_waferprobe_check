# itkpix_waferprobe_check
Scripts for inspecting some ITkPix wafer probing data

## Checkout and Install

In a virtualenv, do:
```
git clone https://github.com/dantrim/itkpix_waferprobe_check.git
cd itkpix_waferprobe_check
pip install -r requirements.txt
```

## Scripts
The script [scripts/dump-k-factors.py](scripts/dump-k-factors.py) takes as input
a wafer probing (HDF5) data file, computes k-factors, and dumps them to CSV
files. Additionally it histograms them and plots the histograms and stores
the plots as PDF files:
```
$ python scripts/dump-k-factors.py --help
Usage: dump-k-factors.py [OPTIONS] INPUT_FILE

  Read wafer probing data and dump CSV files containing k-factor information
  for each chip

Arguments:
  INPUT_FILE  Input wafer probing HDF5 file  [required]
```

## Wafer Probing Data

The wafer probing data is located [here](https://cernbox.cern.ch/index.php/s/c6p5Xrqv4NkfXXj).
You can find the password for accessing the CERNbox link on the [Rd53bTesting Twiki](https://twiki.cern.ch/twiki/bin/viewauth/RD53/RD53BTesting#Wafer_probing).
