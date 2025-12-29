# Computer Vision Assignment: Application of Computer Vision in Thunderstorm Identification & Tracking

## Introduction

This repository implements a compact pipeline for convective storm identification, tracking, and simple nowcasting inspired by TITAN and Enhanced TITAN. It processes 2D radar-like composite images (from Windy), identifies storm cells via thresholding and morphology, estimates motion using TREC variants (COTREC/MTREC), matches cells across time steps (TITAN/eTITAN-style), and produces example visualizations and metrics. Core logic lives in `src/`, notebooks provide runnable demos and evaluation, and `data/` contains small sample inputs, legends, and example outputs. All of the detail work are listed in the report.

## Students Name

| Student Name | Student Id |
| :--- | :---: |
| Tran Quoc Hieu | 2252217 |
| Tran Quoc Trung | 2252859 |

## Project Scope

In this project, we have:
- Reproduced the framework TITAN from paper "Titan: Thunderstorm identification, tracking, analysis, and nowcasting a radar-based methodology" and its improved method - Enhanced TITAN from paper "3d convective storm identification, tracking, and forecasting—an enhanced titan algorithm".
- Utilize the 2 improved versions TREC technique, namely COTREC and MTREC to improve the velocity estimation in tracking steps.
- Load and preprocessing a simple dataset from [Windy](https://www.windy.com/) website.

## Repository 

The structure of this repository include:
```
data/
  images/<region>/ ...          # sample frames per region + optional metadata
  legend/                       # color-to-dBZ and pixel-to-dBZ mappings, use for construct DBZ map
src/
  identification/               # morphology & simple identifiers
  matcher/                      # TITAN/eTITAN matching, COTREC-based flow
  model/                        # reference (e)TITAN implementations
  cores/                        # metrics & movement estimation utilities
    base/                       # Include base object such storm object, storm map and tracking controller
    metrics/                    # Metrics used for evaluate TITAN-based model
    movement_estimate           # TREC, MTREC and COTREC implementation
  utils/                        # IO, preprocessing, legend, polygons
main.ipynb                      # quick run / demo notebook
evaluation_notebooks.ipynb      # evaluation & metrics
README.md
```

## Running code

To run this code, first clone the repository and unzip the file `images.zip`. Then, all codes from the notebook `evaluation_notebooks.ipynb` can be run.