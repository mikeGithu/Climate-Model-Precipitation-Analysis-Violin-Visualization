Climate Model Precipitation Analysis & Violin Visualization
This project performs a comparative analysis of precipitation data from observed datasets and climate model projections (CMIP6). It includes historical and future scenarios (SSP2-4.5, SSP5-8.5) and produces publication-quality violin plots with summary statistics.

🔹 Key Features
Handles multi-dimensional NetCDF data using xarray and converts it into a pandas time series for analysis.

Supports multiple datasets:
Observed precipitation (CHIRPS)
Historical model outputs (raw and bias-corrected)
Future projections (Near, Mid, Far periods; raw and corrected)
Computes evaluation metrics to compare models against observations:
R² (coefficient of determination)
NSE (Nash–Sutcliffe Efficiency)
MSE (Mean Squared Error)
PBIAS (Percent Bias)
Correlation coefficient
Performs temporal alignment to ensure fair comparison (including monthly matching for future projections).

🔹 Workflow
Load datasets
Uses xarray to read NetCDF files.
Aggregates spatial dimensions (latitude/longitude) to produce a single time series per dataset.
Compute metrics
Historical datasets are compared directly with observations.
Future projections are compared using monthly climatology alignment.
Prepare data for visualization
Combines all datasets into a long-format DataFrame suitable for plotting.
Calculates median and mean for each dataset.
Visualization
Creates violin plots to show the distribution of precipitation values per dataset.
Overlays median (black circles) and mean (red diamonds).
Annotates plots with evaluation metrics for each dataset.
Saves figures at high resolution (300 dpi) for publication or reports.

🔹 Tools & Libraries
xarray → Handling multi-dimensional NetCDF climate data
pandas / numpy → Data manipulation and statistics
matplotlib / seaborn → Publication-quality plotting
os → File and directory management
🔹 Purpose
This pipeline allows researchers and analysts to:
Compare climate model outputs against observed precipitation.
Assess the impact of bias correction on model performance.
Explore future climate scenarios (SSP2-4.5 and SSP5-8.5) in a clear, visual manner.
Generate ready-to-use plots for reports, publications, and presentations.
🔹 Outputs
Violin plots showing precipitation distributions across datasets and scenarios.

Summary statistics annotated on plots (R², NSE, MSE, PBIAS, correlation).

Easily extendable to other variables, regions, or climate models.
