# TCD-ML-2

## Overview

University assignment for machine learning, completed as part of a group project. The data is synthetic income data, and the goal is to predict new incomes given this training data. The data was obtained from [here](https://www.kaggle.com/c/tcd-ml-comp-201920-income-pred-group)(accessed 27th November 2019). Note that the training data here has been truncated by about 25% in order to fit into the repository. The final output of the programme can be found in _results.csv_.

## Basic usage

```shell
python3 main.py --train --export
```

Use ```python3 main.py --help``` for more information.

## Requirements

| Name | Version |
|-|-|
| Python | 3.6.9 |
| Scikit-Learn | 0.21.3 |
| Numpy | 1.17.0 |
| Pandas | 0.25.0 |

## Sample data

(sic)

| Instance | Year of Record | Housing Situation | Crime Level in the City of Employement | Work Experience in Current Job [years] | Satisfation with employer | Gender | Age | Country | Size of City | Profession | University Degree | Wears Glasses | Hair Color | Body Height [cm] | Yearly Income in addition to Salary (e.g. Rental Income) | Total Yearly Income [EUR] |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| 1 | 1940 | 0 | 33 | 17 | Unhappy | other | 45 | Afghanistan | 25179 | group head | No | 1 | Black | 182 | 0 EUR | 6182.05 |
| 2 | 1940 | 0 | 25 | 4.9 | Unhappy | female | 17 | Afghanistan | 2278204 | heavy vehicle and mobile equipment service technician | No | 0 | Blond | 172 | 0 EUR | 6819.69 |
| 3 | 1940 | 0 | 34 | 21 | Unhappy | female | 48 | Afghanistan | 822134 | sorter | Bachelor | 0 | Blond | 144 | 0 EUR | 8663.53 |
| 4 | 1940 | 0 | 70 | 18 | Average | female | 42 | Albania | 59477 | quality control senior engineer | No | 1 | Brown | 152 | 0 EUR | 2400.64 |
| 5 | 1940 | 0 | 51 | 8 | Happy | other | 15 | Albania | 23494 | logistician | Master | 1 | Black | 180 | 0 EUR | 2816.18 |
| 6 | 1940 | 0 | 61 | 15 | Average | male | 26 | Albania | 30624 | unix/linux systems lead | Bachelor | 1 | Brown | 212 | 0 EUR | 2572.16 |
| 7 | 1940 | 0 | 58 | 12 | Average | male | 22 | Albania | 288022 | purchasing agent | Bachelor | 1 | Black | 181 | 0 EUR | 3336.93 |
| 8 | 1940 | 0 | 51 | 6.3 | Average | female | 15 | Albania | 1595318 | quality management specialist | Bachelor | 0 | #N/A | 161 | 0 EUR | 3679.14 |
| 9 | 1940 | 0 | 68 | 15 | Happy | male | 37 | Albania | 82114 | investment officer | Bachelor | 1 | Black | 168 | 0 EUR | 2666.37 |
| 10 | 1940 | 0 | 60 | 13 | Average | #N/A | 25 | Albania | 2064899 | rigger | Bachelor | 0 | Brown | 186 | 0 EUR | 3898.08 |
| 11 | 1940 | 0 | 64 | 14 | Average | male | 30 | Albania | 1486936 | pumping station operator | Master | 1 | Black | 179 | 0 EUR | 4895.97 |
| 12 | 1940 | 0 | 77 | 22 | Happy | male | 60 | Albania | 59209 | industrial production manager | No | 0 | Brown | 199 | 0 EUR | 2244.96 |
| 13 | 1940 | 0 | 79 | 26 | Happy | other | 67 | Albania | 26249 | permit records assistant | Bachelor | 1 | Blond | 188 | 0 EUR | 2960.58 |
| 14 | 1940 | 0 | 51 | 8 | Happy | male | 15 | Albania | 10724 | janitorial worker | No | 0 | Blond | 196 | 0 EUR | 1958.56 |
| 15 | 1940 | 0 | 56 | 10 | Somewhat Happy | female | 20 | Albania | 1118504 | policeman | Bachelor | 1 | Black | 159 | 0 EUR | 3812.35 |
| 16 | 1940 | 0 | 80 | 29 | Average | male | 72 | Albania | 91154 | registered nurse | #N/A | 1 | Brown | 203 | 0 EUR | 1930.5 |
| 17 | 1940 | 0 | 66 | 14 | Happy | male | 34 | Albania | 702029 | information security identity & access manager | Bachelor | 0 | Black | 212 | 0 EUR | 3506.28 |
| 18 | 1940 | 0 | 74 | 23 | Average | other | 53 | Albania | 1260228 | senior project officer | #N/A | 0 | Blond | 172 | 0 EUR | 2824.78 |
| 19 | 1940 | 0 | 58 | 11 | Somewhat Happy | male | 22 | Albania | 1390382 | packager | Master | 1 | Blond | 174 | 0 EUR | 4625.05 |
| 20 | 1940 | 0 | 59 | 12 | Somewhat Happy | male | 23 | Albania | 1961433 | parks enforcement patrol | PhD | 0 | Red | 219 | 0 EUR | 5290.37 |
| 21 | 1940 | 0 | 63 | 14 | Average | female | 29 | Albania | 1935 | real estate broker | Master | 1 | Brown | 188 | 0 EUR | 2862.88 |