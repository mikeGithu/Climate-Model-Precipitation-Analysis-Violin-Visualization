#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[66]:


os.chdir(r"D:\New Jouranl Drought\CMIP6")
cwd = os.getcwd()
cwd


# In[67]:


#Observed 
#Hist_Raw 
#Hist_Corr
#Near_Raw
#Near_Corr
#Mid_Raw 
#Mid_Corr
#Far_Raw 
#Far_Corr


# In[68]:


# ======== FONT SETTINGS ========
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"  # Make all text bold


# In[69]:


dt = xr.open_dataset(r"D:\New Jouranl Drought\CMIP6\SSP585\In_pr_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20190116-21001216.nc")
dt


# In[70]:


# ======== FILES & SETTINGS ========
files_vars = {
    "Observed":(r"D:\New Jouranl Drought\PCP\CHIRPS_Genale_total_precipitation_month_0.25x0.25_africa_1983_2014_v2.0.nc", "pr"),
    "Hist_Raw":(r"D:\New Jouranl Drought\CMIP6\Historical\6b46be8f417f237a87f23e90871546fe\pr_In2_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_19830116-20141216.nc", "pr_mm_month"),
    "Hist_Corr":(r"D:\New Jouranl Drought\CMIP6\Historical\corr_pr_In_2_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_19830116-20141216.nc", "pr_corrected"),
    "Near_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Near_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20200116-20491216.nc", "pr_corrected"),
    "Mid_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Mid_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20410116-20701216.nc", "pr_corrected"),
    "Far_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Far_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP245\pr_In_Amon_CNRM-CM6-1_ssp245_r1i1p1f2_gr_20710116-21001216.nc", "pr_corrected"),
}

time_dim = "time"
output_jpg = r"D:\New Jouranl Drought\CMIP6\precip_ssp245_metrics_violin.jpg"


# In[71]:


for name, (path, _) in files_vars.items():
    if not os.path.isfile(path):
        print(f"❌ Missing file: {name} -> {path}")
    else:
        print(f"✅ Found file: {name}")


# In[72]:


# ======== FUNCTIONS ========
def load_series(path, var, time_dim="time"):
    ds = xr.open_dataset(path)
    da = ds[var]
    extra_dims = [d for d in da.dims if d != time_dim]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return da.to_series().dropna()

def align_and_stats(ref, arr):
    joined = pd.concat([ref, arr], axis=1, join="inner").dropna()
    if joined.empty:
        return np.nan, np.nan, np.nan, np.nan
    y_true, y_pred = joined.iloc[:, 0], joined.iloc[:, 1]
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mse = np.mean((y_true - y_pred)**2)
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    nse = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return r2, mse, pbias, nse

# ======== LOAD DATA ========
datasets = {}
for name, (path, varname) in files_vars.items():
    datasets[name] = load_series(path, varname, time_dim)


# In[73]:


# ======== FUNCTIONS ========
def load_series(path, var, time_dim="time"):
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return pd.Series(dtype=float)
    ds = xr.open_dataset(path)
    da = ds[var]
    extra_dims = [d for d in da.dims if d != time_dim]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return da.to_series().dropna()

def compute_metrics(y_true, y_pred):
    """Compute R², NSE, MSE, PBIAS, CORR."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return [np.nan] * 5
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mse = np.mean((y_true - y_pred)**2)
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    nse = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    return r2, mse, pbias, nse, corr

def align_and_stats(ref, arr):
    joined = pd.concat([ref, arr], axis=1, join="inner").dropna()
    return compute_metrics(joined.iloc[:, 0], joined.iloc[:, 1])

def monthly_match_and_stats(obs_series, fut_series):
    obs_df = pd.DataFrame({"Value": obs_series, "Month": obs_series.index.month})
    fut_df = pd.DataFrame({"Value": fut_series, "Month": fut_series.index.month})
    month_map = obs_df.groupby("Month")["Value"].mean()
    fut_df["Obs_Mean"] = fut_df["Month"].map(month_map)
    joined = fut_df.dropna()
    return compute_metrics(joined["Obs_Mean"], joined["Value"])

# ======== LOAD DATA ========
datasets = {}
for name, (path, varname) in files_vars.items():
    datasets[name] = load_series(path, varname, time_dim)


# In[74]:


# ======== PERIODS ========
periods = {
    "Observed": ("1983-01-01", "2013-12-31"),
    "Hist_Raw": ("1983-01-01", "2013-12-31"),
    "Hist_Corr": ("1983-01-01", "2013-12-31"),
    "Near_Raw": ("2020-01-01", "2049-12-31"),
    "Near_Corr": ("2020-01-01", "2049-12-31"),
    "Mid_Raw": ("2041-01-01", "2070-12-31"),
    "Mid_Corr": ("2041-01-01", "2070-12-31"),
    "Far_Raw": ("2071-01-01", "2100-12-31"),
    "Far_Corr": ("2071-01-01", "2100-12-31"),
}


# In[75]:


# Slice datasets
for name, (start, end) in periods.items():
    datasets[name] = datasets[name].loc[start:end]

# ======== STATS ========
obs = datasets["Observed"]
stats_df = []
for name, ser in datasets.items():
    if name == "Observed":
        continue
    if "Hist" in name:
        r2, mse, pbias, nse, corr = align_and_stats(obs, ser)
    else:
        r2, mse, pbias, nse, corr = monthly_match_and_stats(obs, ser)
    stats_df.append({
        "Dataset": name,
        "R2": r2, "MSE": mse, "PBIAS": pbias, "NSE": nse, "CORR": corr
    })
stats_df = pd.DataFrame(stats_df)

# ======== LONG DF FOR VIOLIN ========
df_long = pd.concat(datasets, axis=1).melt(var_name="Dataset", value_name="Value").dropna()

# ======== PLOT ========
sns.set(style="whitegrid", font_scale=1)
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

sns.violinplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, inner=None, scale="width", palette="Set2"
)
sns.pointplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, join=False, estimator=np.median,
    color="black", markers="o", scale=0.6
)
sns.pointplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, join=False, estimator=np.mean,
    color="red", markers="D", scale=0.5
)

# Add stats above violins
ymax = df_long["Value"].max()
for i, name in enumerate(df_long["Dataset"].unique()):
    row = stats_df[stats_df["Dataset"] == name]
    if not row.empty:
        s = row.iloc[0]
        text = (
            f"R²={s['R2']:.2f}\n"
            f"NSE={s['NSE']:.2f}\n"
            f"MSE={s['MSE']:.2f}\n"
            f"PBias={s['PBIAS']:.1f}%\n"
            f"Corr={s['CORR']:.2f}"
        )
        ax.text(i, ymax + 0.05*ymax, text,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

ax.set_title("Precipitation Distribution of SSP2-4.5", fontsize=16, fontweight='bold', pad=30)
ax.set_xlabel("")
ax.set_ylabel("precipitation (mm/month)")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Plot saved to {output_jpg}")


# In[32]:


# ======== FILES & SETTINGS ========
files_vars = {
    "Observed":(r"D:\New Jouranl Drought\PCP\CHIRPS_Genale_total_precipitation_month_0.25x0.25_africa_1983_2014_v2.0.nc", "pr"),
    "Hist_Raw":(r"D:\New Jouranl Drought\CMIP6\Historical\pr_In_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_19830116-20141216.nc", "pr_mm_month"),
    "Hist_Corr":(r"D:\New Jouranl Drought\CMIP6\Historical\corr_pr_In_2_Amon_CNRM-CM6-1_historical_r1i1p1f2_gr_19830116-20141216.nc", "pr_corrected"),
    "Near_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In3_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Near_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20200116-20491216.nc", "pr_corrected"),
    "Mid_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In3_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Mid_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20410116-20701216.nc", "pr_corrected"),
    "Far_Raw":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In3_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_20190116-21001216.nc", "pr_mm_month"),
    "Far_Corr":(r"D:\New Jouranl Drought\CMIP6\SSP585\pr_In_Amon_CNRM-CM6-1_ssp585_r1i1p1f2_gr_2071116-21001216.nc", "pr_corrected"),
}

time_dim = "time"
output_jpg = r"D:\New Jouranl Drought\CMIP6\precip_ssp485_metrics_violin.jpg"


# In[33]:


# ======== FUNCTIONS ========
def load_series(path, var, time_dim="time"):
    ds = xr.open_dataset(path)
    da = ds[var]
    extra_dims = [d for d in da.dims if d != time_dim]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return da.to_series().dropna()

def align_and_stats(ref, arr):
    joined = pd.concat([ref, arr], axis=1, join="inner").dropna()
    if joined.empty:
        return np.nan, np.nan, np.nan, np.nan
    y_true, y_pred = joined.iloc[:, 0], joined.iloc[:, 1]
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mse = np.mean((y_true - y_pred)**2)
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    nse = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    return r2, mse, pbias, nse

# ======== LOAD DATA ========
datasets = {}
for name, (path, varname) in files_vars.items():
    datasets[name] = load_series(path, varname, time_dim)


# In[34]:


# ======== FUNCTIONS ========
def load_series(path, var, time_dim="time"):
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return pd.Series(dtype=float)
    ds = xr.open_dataset(path)
    da = ds[var]
    extra_dims = [d for d in da.dims if d != time_dim]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return da.to_series().dropna()

def compute_metrics(y_true, y_pred):
    """Compute R², NSE, MSE, PBIAS, CORR."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return [np.nan] * 5
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    mse = np.mean((y_true - y_pred)**2)
    pbias = 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    nse = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan
    return r2, mse, pbias, nse, corr

def align_and_stats(ref, arr):
    joined = pd.concat([ref, arr], axis=1, join="inner").dropna()
    return compute_metrics(joined.iloc[:, 0], joined.iloc[:, 1])

def monthly_match_and_stats(obs_series, fut_series):
    obs_df = pd.DataFrame({"Value": obs_series, "Month": obs_series.index.month})
    fut_df = pd.DataFrame({"Value": fut_series, "Month": fut_series.index.month})
    month_map = obs_df.groupby("Month")["Value"].mean()
    fut_df["Obs_Mean"] = fut_df["Month"].map(month_map)
    joined = fut_df.dropna()
    return compute_metrics(joined["Obs_Mean"], joined["Value"])

# ======== LOAD DATA ========
datasets = {}
for name, (path, varname) in files_vars.items():
    datasets[name] = load_series(path, varname, time_dim)


# In[35]:


# ======== PERIODS ========
periods = {
    "Observed": ("1983-01-01", "2012-12-31"),
    "Hist_Raw": ("1983-01-01", "2012-12-31"),
    "Hist_Corr": ("1983-01-01", "2012-12-31"),
    "Near_Raw": ("2020-01-01", "2049-12-31"),
    "Near_Corr": ("2020-01-01", "2049-12-31"),
    "Mid_Raw": ("2041-01-01", "2070-12-31"),
    "Mid_Corr": ("2041-01-01", "2070-12-31"),
    "Far_Raw": ("2071-01-01", "2100-12-31"),
    "Far_Corr": ("2071-01-01", "2100-12-31"),
}


# In[36]:


# Slice datasets
for name, (start, end) in periods.items():
    datasets[name] = datasets[name].loc[start:end]

# ======== STATS ========
obs = datasets["Observed"]
stats_df = []
for name, ser in datasets.items():
    if name == "Observed":
        continue
    if "Hist" in name:
        r2, mse, pbias, nse, corr = align_and_stats(obs, ser)
    else:
        r2, mse, pbias, nse, corr = monthly_match_and_stats(obs, ser)
    stats_df.append({
        "Dataset": name,
        "R2": r2, "MSE": mse, "PBIAS": pbias, "NSE": nse, "CORR": corr
    })
stats_df = pd.DataFrame(stats_df)

# ======== LONG DF FOR VIOLIN ========
df_long = pd.concat(datasets, axis=1).melt(var_name="Dataset", value_name="Value").dropna()

# ======== PLOT ========
sns.set(style="whitegrid", font_scale=1)
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

sns.violinplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, inner=None, scale="width", palette="Set2"
)
sns.pointplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, join=False, estimator=np.median,
    color="black", markers="o", scale=0.6
)
sns.pointplot(
    data=df_long, x="Dataset", y="Value",
    ax=ax, join=False, estimator=np.mean,
    color="red", markers="D", scale=0.5
)

# Add stats above violins
ymax = df_long["Value"].max()
for i, name in enumerate(df_long["Dataset"].unique()):
    row = stats_df[stats_df["Dataset"] == name]
    if not row.empty:
        s = row.iloc[0]
        text = (
            f"R²={s['R2']:.2f}\n"
            f"NSE={s['NSE']:.2f}\n"
            f"MSE={s['MSE']:.2f}\n"
            f"PBias={s['PBIAS']:.1f}%\n"
            f"Corr={s['CORR']:.2f}"
        )
        ax.text(i, ymax + 0.05*ymax, text,
                ha='center', va='bottom', fontsize=7,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

ax.set_title("Precipitation Distribution of SSP5-8.5", fontsize=16, fontweight='bold', pad=30)
ax.set_xlabel("")
ax.set_ylabel("precipitation (mm/month)")
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Plot saved to {output_jpg}")

