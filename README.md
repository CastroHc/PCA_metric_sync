# PCA-Based Synchronization Analysis of Coupled Oscillators

This repository contains the Python code for analyzing synchronization phenomena in coupled Rossler oscillators using Principal Component Analysis (PCA). The methodology is based on the research presented in the following article:

**Castro, H.C. and Aguirre, L.A. (2025).**  
*The Latent Variable Subspace in Synchronization Problems.*  
Journal of Control, Automation and Electrical Systems, DOI: (https://doi.org/10.1007/s40313-025-01179-0)

## Overview

The code simulates a pair of coupled Rossler systems across a range of coupling strengths. It applies PCA to the combined state data to quantify shared dynamics through the explained variance ratio, serving as an indicator of synchronization. Additionally, it computes phase coherence as a comparative measure.

This approach provides a data-driven alternative to traditional phase-based synchronization metrics, particularly useful when phase extraction is challenging.

## Contents

- `latent_space_analysis_synchronization.py`  
  Main Python script implementing the simulation, PCA analysis, and visualization.

## Usage

1. Ensure you have the required Python libraries installed:  
   - numpy  
   - pandas  
   - scikit-learn  
   - matplotlib  
   - numba
   - control

2. Run the script:  
   ```bash
   python latent_space_analysis_synchronization.py
