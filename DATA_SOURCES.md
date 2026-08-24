# Data sources and integrity record

The benchmark code is MIT-licensed. The datasets are third-party research
artifacts and remain subject to their original terms; they are not covered by
the software licence. The public repository need not redistribute the raw
files: `scripts/download_data.py` retrieves the exact filenames from the
documented sources, and the SHA-256 values below verify the inputs used for the
reported experiments.

## BATADAL

Primary project/citation: R. Taormina et al., *The Battle of the Attack
Detection Algorithms*, Journal of Water Resources Planning and Management 144
(8), 04018048 (2018), DOI: 10.1061/(ASCE)WR.1943-5452.0000969.

Retrieval mirror maintained with the BATADAL project materials:

- `https://raw.githubusercontent.com/scy-phy/www.batadal.net/master/data/BATADAL_dataset03.csv`
- `https://raw.githubusercontent.com/scy-phy/www.batadal.net/master/data/BATADAL_dataset04.csv`

## Gas Pipeline

Primary citation: T. Morris and W. Gao, *Industrial control system network
traffic data sets to facilitate intrusion detection system research*, Critical
Infrastructure Protection VIII, Springer (2014), pp. 65–78.

Retrieval URL used by the script:

- `https://raw.githubusercontent.com/Rocionightwater/ML-NIDS-for-SCADA/master/data/IanArffDataset.arff`

## HAI 22.04

Primary repository: `https://github.com/icsdataset/hai`. The benchmark uses the
HAI 22.04 `train1.csv` and `test1.csv` files.

Retrieval base used by the script:

- `https://media.githubusercontent.com/media/icsdataset/hai/master/hai-22.04/`

## SHA-256 values for the analysed files

| File | SHA-256 |
|---|---|
| `data/batadal/BATADAL_dataset03.csv` | `8CA6CB851242254D2605B5A53BA3BA5009E2213A545D65396615F1CE0B426E1B` |
| `data/batadal/BATADAL_dataset04.csv` | `4746BEAB2CFCDB5E68C7FA197A7522AECB70189D61DCC3EB1F870D2CE387068B` |
| `data/gas/IanArffDataset.arff` | `8E1E9804020CAD3E6870C24D232B83DC21C3A5F877B1EA618B8844C668BB18A7` |
| `data/hai/train1.csv` | `63383690FFC5344BA68A019C7000854A4D341A22D743AC035F5207072A197B33` |
| `data/hai/test1.csv` | `19627F5DAC40C3A5039E7468CFD6C6E4CCD37A85055671E8C349B6BCE5F50ED1` |

Accessed and verified: 24 August 2026.

