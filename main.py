#!/usr/bin/env python3

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker as mticker

# Tkinter GUI wymaga środowiska graficznego
if not (os.environ.get("DISPLAY") or sys.platform.startswith("win")):
    print("GUI Tk wymagane.", file=sys.stderr)
    sys.exit(1)
matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ───────────  OBLICZENIA  ───────────
def calc_alpha(df: pd.DataFrame,
               wl: str,
               Tcol: str,
               Rcol: str | None = None,
               d_cm: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["Energy_eV"] = 1240.0 / pd.to_numeric(df[wl], errors="coerce")
    T = pd.to_numeric(df[Tcol], errors="coerce") / 100
    if Rcol:
        R = pd.to_numeric(df[Rcol], errors="coerce") / 100
        with np.errstate(divide="ignore", invalid="ignore"):
            df["alpha"] = np.log(T / (1 - R)) / d_cm
            df["T/(1-R)_or_T"] = np.where((1 - R) != 0, T / (1 - R), 0.0)
    else:
        with np.errstate(divide="ignore"):
            df["alpha"] = -np.log(T) / d_cm
            df["T/(1-R)_or_T"] = T
    return df


def tauc_y(alpha: np.ndarray, E: np.ndarray, n: float) -> np.ndarray:
    return (alpha * E) ** n


# ═══════════════════  GUI  ═══════════════════
class TaucApp(tk.Tk):
    W, H = 550, 260

    def __init__(self) -> None:
        super().__init__()
        self.title("Tauc / Urbach analyser v8")
        self.geometry("1700x960")

        # dane
        self.df_T: Optional[pd.DataFrame] = None
        self.df_R: Optional[pd.DataFrame] = None
        self.df_sel: Optional[pd.DataFrame] = None
        self.col_T: Optional[str] = None
        self.col_R: Optional[str] = None
        self.mode: str = "tauc"
        self.fits: List[dict] = []            # << dopasowania
        self.grid_on = False

        # grubość
        self.d_nm_var = tk.StringVar(value="")
        self.d_cm_var = tk.StringVar(value="—")
        self.d_nm_var.trace_add("write", self._update_thickness_cm)

        # grafika pomocnicza
        self.scatter_fits: List[matplotlib.collections.PathCollection] = []
        self._preview_objs: Tuple[
            Optional[matplotlib.lines.Line2D],
            Optional[matplotlib.collections.PathCollection]
        ] = (None, None)

        self._menu(); self._layout()

    # ─ MENU ─
    def _menu(self):
        m = tk.Menu(self)
        fm = tk.Menu(m, tearoff=0)
        for label, cmd in (("Load T…", self.load_T),
                           ("Load R…", self.load_R)):
            fm.add_command(label=label, command=cmd)
        fm.add_separator()
        fm.add_command(label="Export CSV…", command=self.export_csv)
        fm.add_command(label="Save plot…", command=self.save_plot)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.destroy)
        m.add_cascade(label="File", menu=fm)
        self.config(menu=m)

    # ─ LAYOUT ─
    def _panel(self, col: int, title: str):
        f = ttk.LabelFrame(self, text=title, padding=4,
                           width=self.W, height=self.H)
        f.grid(row=0, column=col, padx=4, pady=4, sticky="n")
        f.grid_propagate(False); f.columnconfigure(0, weight=1)
        tree = ttk.Treeview(f, show="headings", height=7)
        tree.grid(row=0, column=0, sticky="nsew")
        hsb = ttk.Scrollbar(f, orient="horizontal", command=tree.xview)
        vsb = ttk.Scrollbar(f, orient="vertical", command=tree.yview)
        hsb.grid(row=1, column=0, sticky="ew"); vsb.grid(row=0, column=1, sticky="ns")
        tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)
        return f, tree

    def _layout(self):
        self.columnconfigure((0, 1, 2), weight=0)
        self.rowconfigure(0, weight=0, minsize=self.H)
        self.rowconfigure(1, weight=1)

        fT, self.tree_T = self._panel(0, "Raw T-data")
        fR, self.tree_R = self._panel(1, "Raw R-data")
        fS, self.tree_S = self._panel(2, "Selected")

        # T controls
        ctrlT = ttk.Frame(fT); ctrlT.grid(row=2, column=0, pady=4)
        ttk.Label(ctrlT, text="λ").pack(side=tk.LEFT)
        self.cmb_wl_T = ttk.Combobox(ctrlT, width=14, state="readonly")
        self.cmb_wl_T.pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrlT, text="T").pack(side=tk.LEFT)
        self.cmb_T = ttk.Combobox(ctrlT, width=14, state="readonly")
        self.cmb_T.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrlT, text="Apply T →", command=self.apply_T)\
            .pack(side=tk.LEFT, padx=6)

        # R controls
        ctrlR = ttk.Frame(fR); ctrlR.grid(row=2, column=0, pady=4)
        ttk.Label(ctrlR, text="λ").pack(side=tk.LEFT)
        self.cmb_wl_R = ttk.Combobox(ctrlR, width=14, state="readonly")
        self.cmb_wl_R.pack(side=tk.LEFT, padx=2)
        ttk.Label(ctrlR, text="R").pack(side=tk.LEFT)
        self.cmb_R = ttk.Combobox(ctrlR, width=14, state="readonly")
        self.cmb_R.pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrlR, text="Apply R →", command=self.apply_R)\
            .pack(side=tk.LEFT, padx=6)

        # Selected controls
        ctrlS = ttk.Frame(fS); ctrlS.grid(row=2, column=0, pady=2, sticky="w")
        self.tr_var = tk.StringVar(value="direct")
        for t, v in (("Direct", "direct"), ("Indirect", "indirect")):
            ttk.Radiobutton(ctrlS, text=t, variable=self.tr_var, value=v)\
                .pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrlS, text="Plot Tauc", command=self.plot_tauc)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrlS, text="Plot Urbach", command=self.plot_urbach)\
            .pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrlS, text="Grid", command=self.toggle_grid)\
            .pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrlS, text="Clear", command=self.clear_sel)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(ctrlS, text="Clear fits", command=self.clear_fits)\
            .pack(side=tk.LEFT, padx=4)

        # thickness
        thick = ttk.Frame(fS); thick.grid(row=3, column=0, pady=2, sticky="w")
        ttk.Label(thick, text="d [nm]:").pack(side=tk.LEFT)
        ttk.Entry(thick, width=8, textvariable=self.d_nm_var)\
            .pack(side=tk.LEFT, padx=2)
        ttk.Label(thick, text="( =").pack(side=tk.LEFT)
        ttk.Label(thick, textvariable=self.d_cm_var).pack(side=tk.LEFT)
        ttk.Label(thick, text="cm )").pack(side=tk.LEFT)

        # plot area
        pf = ttk.LabelFrame(self, text="Plot area", padding=4)
        pf.grid(row=1, columnspan=3, sticky="nsew", padx=4, pady=4)
        pf.rowconfigure(0, weight=1); pf.columnconfigure(0, weight=1)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=pf)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, pf, pack_toolbar=False)
        toolbar.update(); toolbar.grid(row=1, column=0, sticky="ew")
        self.span: Optional[SpanSelector] = None

    # ─ helper: read ─
    def _read(self, p: str) -> pd.DataFrame:
        p = Path(p)
        return pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} \
               else pd.read_csv(p, sep=None, engine="python")

    # ─ Load T / R ─
    def load_T(self):
        p = filedialog.askopenfilename(filetypes=[("CSV/XLSX", "*.csv *.xlsx *.xls")])
        if not p: return
        try: self.df_T = self._read(p)
        except Exception as e:
            messagebox.showerror("Load T", str(e)); return
        self._fill_tree(self.tree_T, self.df_T)
        cols = list(self.df_T.columns)
        self.cmb_wl_T["values"] = cols; self.cmb_wl_T.set("")
        self.cmb_T["values"] = cols; self.cmb_T.set("")

    def load_R(self):
        p = filedialog.askopenfilename(filetypes=[("CSV/XLSX", "*.csv *.xlsx *.xls")])
        if not p: return
        try: self.df_R = self._read(p)
        except Exception as e:
            messagebox.showerror("Load R", str(e)); return
        self._fill_tree(self.tree_R, self.df_R)
        cols = list(self.df_R.columns)
        self.cmb_wl_R["values"] = cols; self.cmb_wl_R.set("")
        self.cmb_R["values"] = cols; self.cmb_R.set("")

    # ─ merge ─
    def _merge(self, sub: pd.DataFrame):
        self.df_sel = sub if self.df_sel is None \
                      else pd.merge(self.df_sel, sub, on="lambda", how="outer")
        self.df_sel.sort_values("lambda", ascending=False, inplace=True)
        self._fill_tree(self.tree_S, self.df_sel)

    def apply_T(self):
        if self.df_T is None:
            messagebox.showinfo("Apply T", "Najpierw wczytaj plik T."); return
        wl, col = self.cmb_wl_T.get(), self.cmb_T.get()
        if not wl or not col:
            messagebox.showinfo("Apply T", "Wskaż kolumny λ i T."); return
        self.col_T = f"{col}_T"
        sub = self.df_T[[wl, col]].copy().rename(
            columns={wl: "lambda", col: self.col_T})
        self._merge(sub)

    def apply_R(self):
        if self.df_R is None:
            messagebox.showinfo("Apply R", "Najpierw wczytaj plik R."); return
        wl, col = self.cmb_wl_R.get(), self.cmb_R.get()
        if not wl or not col:
            messagebox.showinfo("Apply R", "Wskaż kolumny λ i R."); return
        self.col_R = f"{col}_R"
        sub = self.df_R[[wl, col]].copy().rename(
            columns={wl: "lambda", col: self.col_R})
        self._merge(sub)

    # ─ thickness helpers ─
    def _d_cm(self) -> float:
        try: return float(self.d_nm_var.get()) * 1e-7
        except ValueError: return 1.0

    def _update_thickness_cm(self, *_):
        try:
            self.d_cm_var.set(f"{float(self.d_nm_var.get())*1e-7:.3e}")
        except ValueError:
            self.d_cm_var.set("—")

    # ─ prepare ─
    def _prepare(self) -> Optional[pd.DataFrame]:
        if self.df_sel is None:
            messagebox.showinfo("Plot", "Brak danych."); return None
        if not self.col_T:
            messagebox.showinfo("Plot", "Nie wskazano kolumny T."); return None

        df = self.df_sel.copy()
        df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
        df.dropna(subset=["lambda", self.col_T], inplace=True)

        return calc_alpha(df, "lambda", self.col_T, self.col_R, self._d_cm())\
               if not df.empty else None

    # ─ DRAW helpers ─
    def _draw(self, x, y, title, xl, yl):
        self.ax.clear()
        self.ax.scatter(x, y, s=9, color="tab:orange")
        self.ax.set_xlabel(xl); self.ax.set_ylabel(yl); self.ax.set_title(title)
        self.ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        self.ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.01))
        self.ax.tick_params(axis="x", which="minor", length=3)
        self.ax.set_xlim(left=0)
        self.scatter_fits.clear()

        if self.span: self.span.disconnect_events()
        self.span = SpanSelector(
            self.ax, self._on_span_final, "horizontal",
            props=dict(alpha=0.2), onmove_callback=self._on_span_preview,
            useblit=True)
        self.canvas.draw_idle()

    # ─ preview ─
    def _on_span_preview(self, x0, x1):
        line, pt = self._preview_objs
        if line: line.remove()
        if pt:   pt.remove()

        lo, hi = sorted((x0, x1))
        mask = (self.curX >= lo) & (self.curX <= hi)
        if mask.sum() < 2:
            self._preview_objs = (None, None)
            self.ax.legend().remove() if self.ax.get_legend() else None
            self.canvas.draw_idle(); return

        a, b = np.polyfit(self.curX[mask], self.curY[mask], 1)
        y_pred = a*self.curX[mask] + b
        r2 = 1 - np.sum((self.curY[mask]-y_pred)**2) / \
                 np.sum((self.curY[mask]-np.mean(self.curY[mask]))**2)

        if self.mode == "tauc":
            Eg = -b / a
            xx = np.linspace(min(lo, Eg), max(hi, Eg), 200)
            label = f"Eg≈{Eg:.3f} eV | R²={r2:.3f}"
            line_prev, = self.ax.plot(xx, a*xx + b, "--",
                                      color="steelblue", lw=1, label=label)
            pt_prev = self.ax.scatter([Eg], [0], color="red", zorder=5)
            self._preview_objs = (line_prev, pt_prev)
        else:
            Eu = 1000/a if a else np.nan
            xx = np.linspace(lo, hi, 200)
            label = f"Eu≈{Eu:.1f} meV | R²={r2:.3f}"
            line_prev, = self.ax.plot(xx, a*xx + b, "--",
                                      color="steelblue", lw=1, label=label)
            self._preview_objs = (line_prev, None)

        self.ax.legend(loc="best", frameon=False)
        self.canvas.draw_idle()

    # ─ final ─
    def _on_span_final(self, x0, x1):
        for obj in self._preview_objs:
            if obj: obj.remove()
        self._preview_objs = (None, None)
        self.ax.legend().remove() if self.ax.get_legend() else None

        lo, hi = sorted((x0, x1))
        mask = (self.curX >= lo) & (self.curX <= hi)
        if mask.sum() < 2: return

        a, b = np.polyfit(self.curX[mask], self.curY[mask], 1)
        y_pred = a*self.curX[mask] + b
        r2 = 1 - np.sum((self.curY[mask]-y_pred)**2) / \
                 np.sum((self.curY[mask]-np.mean(self.curY[mask]))**2)

        if self.mode == "tauc":
            Eg = -b / a
            xx = np.linspace(min(lo, Eg), max(hi, Eg), 200)
            label = f"Eg={Eg:.3f} eV | R²={r2:.3f}"
            self.ax.plot(xx, a*xx + b, label=label)
            pt_fin = self.ax.scatter([Eg], [0], color="red", zorder=5)
            self.scatter_fits.append(pt_fin)
            self.fits.append(dict(slope=a, inter=b, Eg=Eg, R2=r2))
        else:
            Eu = 1000/a if a else np.nan
            xx = np.linspace(lo, hi, 200)
            label = f"Eu={Eu:.1f} meV | R²={r2:.3f}"
            self.ax.plot(xx, a*xx + b, label=label)
            self.fits.append(dict(slope=a, inter=b, Eu_meV=Eu, R2=r2))

        self.ax.legend(loc="best", frameon=False)
        self.canvas.draw_idle()

    # ─ plot cmds ─
    def plot_tauc(self):
        dfp = self._prepare(); self.mode = "tauc"
        if dfp is None: return
        n = 0.5 if self.tr_var.get()=="direct" else 2.0
        E, alpha = dfp["Energy_eV"].values, dfp["alpha"].values
        mask = np.isfinite(E)
        self.curX = E[mask]
        self.curY = tauc_y(alpha[mask], self.curX, n)
        self._draw(self.curX, self.curY, f"Tauc ({self.tr_var.get()})",
                   "Energy [eV]", f"(αE)^{n}")
        self.fits.clear()

    def plot_urbach(self):
        dfp = self._prepare(); self.mode = "urbach"
        if dfp is None: return
        lnα = np.log(-dfp["alpha"].where(dfp["alpha"] < 0))
        mask = np.isfinite(lnα)
        self.curX = dfp["Energy_eV"][mask].values
        self.curY = lnα[mask].values
        self._draw(self.curX, self.curY, "Urbach", "Energy [eV]", "ln α")
        self.fits.clear()

    # ─ misc buttons ─
    def toggle_grid(self):
        self.grid_on = not self.grid_on
        self.ax.grid(self.grid_on); self.canvas.draw_idle()

    def clear_sel(self):
        self.df_sel = None; self.col_T = self.col_R = None
        self.tree_S.delete(*self.tree_S.get_children())

    def clear_fits(self):
        self.fits.clear()
        for line in list(self.ax.get_lines()): line.remove()
        for pt in self.scatter_fits: pt.remove()
        self.scatter_fits.clear()
        self.ax.legend().remove() if self.ax.get_legend() else None
        self.canvas.draw_idle()

    # ─ EXPORT CSV ─
    def export_csv(self):
        if self.df_sel is None:
            messagebox.showinfo("Export", "Brak danych."); return
        p = filedialog.asksaveasfilename(defaultextension=".csv")
        if not p: return

        df_base = self.df_sel.copy()
        df_base["d_cm"] = self._d_cm()

        prep = self._prepare()
        if prep is not None:
            df_base = df_base.merge(
                prep[["Energy_eV", "T/(1-R)_or_T", "alpha"]],
                left_index=True, right_index=True, how="left")

        if self.mode == "tauc" and hasattr(self, "curY"):
            df_base["y_tauc"] = np.nan
            df_base["y_tauc"] = self.curY
        elif self.mode == "urbach" and hasattr(self, "curY"):
            df_base["ln_alpha"] = np.nan
            df_base.loc[df_base["Energy_eV"].isin(self.curX), "ln_alpha"]\
                = self.curY

        first = ["lambda"]
        if self.col_T: first.append(self.col_T)
        if self.col_R: first.append(self.col_R)
        order = first + ["Energy_eV", "T/(1-R)_or_T", "alpha"]
        order += [c for c in ("y_tauc", "ln_alpha") if c in df_base.columns]
        rest = [c for c in df_base.columns if c not in order]
        df_base = df_base[order + rest]

        # ❶ dane + ❷ pusta linia + ❸ parametry regresji
        frames = [df_base]
        if self.fits:
            frames += [pd.DataFrame([[]]), pd.DataFrame(self.fits)]
        pd.concat(frames, ignore_index=True).to_csv(p, index=False)
        messagebox.showinfo("Export", f"Saved → {p}")

    def save_plot(self):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("SVG", "*.svg")])
        if p: self.fig.savefig(p, dpi=300, bbox_inches="tight")

    # ─ trees ─
    def _fill_tree(self, tree: ttk.Treeview, df: pd.DataFrame):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        for c in df.columns:
            tree.heading(c, text=c); tree.column(c, width=95, stretch=False)
        for _, row in df.head(200).iterrows():
            tree.insert("", "end", values=list(row))


# ─ main ─
if __name__ == "__main__":
    TaucApp().mainloop()
