# traffik
Traffik - video summarization artificial dataset 🚗

Install `pytorch` separately and then install the other dependencies:

```bash
pip install -r requirements.txt
```


Install the package locally: 

```bash
pip install -e .
```

Generate your dataset.

```
Usage: python gen.py [OPTIONS]

Options:
  --source TEXT  Path of the directory that contains videos and CSVs
  --save TEXT    Path where to store the hdf5 dataset file
  --out TEXT     Name of the generated HDF5 dataset file.
  --help         Show this message and exit.
```