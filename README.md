# Tauc / Urbach Plotter App

Interactive Tkinter GUI for rapid Tauc and Urbach analysis with **live preview**, dynamic legend, full data + fit export and sample-thickness support.

---

## Features

| ✔ | Description |
|---|-------------|
| 🔎 **Real-time preview** | While you drag the SpanSelector, the regression line, Eg/Eu point and legend update continuously (with _R²_). |
| 📐 **Full CSV export**   | Adds calculated columns e.g. (`alpha`,`y_tauc`,`ln_alpha`) **and** a table of fit parameters (`slope`, `inter`, `Eg`, `R2` / `Eu_meV`, `R2`). |
| 💾 **Save figure**       | Export PNG or SVG (300 dpi, tight bbox). |

---

## Requirements

| Tool | Tested version |
|------|----------------|
| Python | ≥ 3.9 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.0 |
| matplotlib | ≥ 3.8 |

> **Tkinter** ships with the official CPython installers.  
> On Linux you may need `sudo apt install python3-tk`.

Install everything in one go:

```bash
pip install -r requirements.txt
```

*(The `requirements.txt` file is included in the repository.)*

---

## Usage · Step‑by‑step

| UI element | What it does | Notes |
|------------|--------------|-------|
| **Load T…** | Open a CSV / XLS(X) that contains transmission data (in %). | Any column names are fine – you pick them next. |
| **Load R…** | *(Optional)* load a file with reflection data (in %). | Skip for pure‑transmission samples. |
| **λ combobox** | Choose the wavelength column (nm). | |
| **T / R combobox** | Choose the percentage column (T or R). | You can load multiple columns one after another – they’ll be merged by λ. |
| **Apply →** | Copies the selected columns into the **Selected** panel. | You can re‑apply after loading more columns. |
| **d [nm]** | Film thickness. Leave blank → defaults to **1 cm**. | Updates the internal `d_cm` label live. |
| **Direct / Indirect** | Select transition type for the Tauc fit. | |
| **Plot Tauc / Plot Urbach** | Draws the scatter plot in the bottom pane. | Switch films? Hit the button again – re‑draws everything. |
| **Grid** | Toggle grid lines. | |
| **Clear** | Wipes the *Selected* table (raw data). | |
| **Clear fits** | Removes all regression lines and Eg/Eu points from the plot. | |
| **Export CSV** | Saves raw + computed columns **and** a table of fit parameters. | Perfect for attaching to a paper or further analysis in Excel/Origin. |
| **Save plot** | PNG / SVG (300 dpi, tight bbox). | |

---

## Quick Start

```bash
python main.py
```

---

## Typical workflow

1. **Prepare a CSV/XLSX**

   ```text
   wavelength_nm, T_%, …
   300, 87.2, …
   310, 86.9, …
   ```
   **and**
   ```text
   wavelength_nm, R_%, …
   300, 9.5, …
   310, 9.8, …
   ```
   Only a wavelength column is mandatory. Percentages must be **0 – 100 %**.

2. Load **T** (and **R**, if available), pick columns, click **Apply →**.  
3. Enter film thickness (`d [nm]`); blank = **1 cm**.  
4. Choose **Direct / Indirect** and press **Plot Tauc** or **Plot Urbach**.  
5. Drag‑select the linear region – preview shows live fit with *Eg/Eu* and *R²*.  
6. Release mouse to lock the fit; repeat for extra segments if needed.  
7. **Export CSV** – raw data + derived columns + fit table appended:  

   | slope | inter | Eg | R2 |
   |-------|-------|----|----|
   | 1.23 × 10⁵ | –1.02 × 10⁵ | 3.27 | 0.998 |

8. **Save plot** PNG or vector SVG.

---

## File format details

| Column | When it appears | Meaning |
|--------|-----------------|---------|
| `Energy_eV` | always | 1240 / λ (nm) |
| `T/(1-R)_or_T` | when T or T & R loaded | Corrected transmittance / transmittance |
| `alpha` | always | Absorption coefficient (cm⁻¹) |
| `y_tauc`   | after a Tauc plot   | (α E)ⁿ – points used for the linear fit |
| `ln_alpha` | after an Urbach plot| ln\|α\| – natural log of the absorption coefficient |                     |
| `d_cm` | always | Film thickness in cm |
| *(Fit table)* | in exported CSV | `slope`, `inter`, `Eg` / `Eu_meV`, `R2` for each stored line |

---

