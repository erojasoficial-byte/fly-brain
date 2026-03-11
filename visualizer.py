"""
Drosophila Brain Activity Visualizer
Interactive GUI to visualize spike propagation from the fly brain simulation.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from pathlib import Path


class FlyBrainVisualizer:
    def __init__(self, root, df):
        self.root = root
        self.root.title("Drosophila Brain - Neural Activity Visualizer")
        self.root.configure(bg='#0a0a0a')
        self.root.state('zoomed')

        self.df = df
        self.neurons = sorted(df['flywire_id'].unique())
        self.neuron_idx = {nid: i for i, nid in enumerate(self.neurons)}
        self.n_neurons = len(self.neurons)
        self.t_max = df['t'].max()
        self.t_min = df['t'].min()

        # Precompute spike times per neuron
        self.spike_times = {}
        for nid in self.neurons:
            self.spike_times[nid] = df[df['flywire_id'] == nid]['t'].values

        # Precompute binned firing rates
        self.bin_size = 5.0  # ms
        self.time_bins = np.arange(0, self.t_max + self.bin_size, self.bin_size)
        self.n_bins = len(self.time_bins) - 1

        # Activity matrix: neurons x time_bins
        self.activity_matrix = np.zeros((self.n_neurons, self.n_bins))
        for nid in self.neurons:
            idx = self.neuron_idx[nid]
            times = self.spike_times[nid]
            hist, _ = np.histogram(times, bins=self.time_bins)
            self.activity_matrix[idx] = hist

        # Sort neurons by total spike count for better visualization
        total_spikes = self.activity_matrix.sum(axis=1)
        self.sort_order = np.argsort(-total_spikes)
        self.activity_sorted = self.activity_matrix[self.sort_order]

        # Animation state
        self.playing = False
        self.current_bin = 0
        self.speed = 50  # ms per frame
        self.anim_id = None

        # Custom colormap: black -> blue -> cyan -> white
        colors = ['#0a0a0a', '#0d1b4a', '#1b4f8a', '#00b4d8', '#48cae4', '#90e0ef', '#ffffff']
        self.cmap = LinearSegmentedColormap.from_list('neural', colors, N=256)

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#0a0a0a')
        style.configure('Dark.TLabel', background='#0a0a0a', foreground='#48cae4',
                         font=('Consolas', 11))
        style.configure('Title.TLabel', background='#0a0a0a', foreground='#00b4d8',
                         font=('Consolas', 16, 'bold'))
        style.configure('Stat.TLabel', background='#0a0a0a', foreground='#90e0ef',
                         font=('Consolas', 12))
        style.configure('Dark.TButton', font=('Consolas', 11))

        # Header
        header = ttk.Frame(self.root, style='Dark.TFrame')
        header.pack(fill='x', padx=10, pady=(10, 0))

        ttk.Label(header, text="DROSOPHILA MELANOGASTER - BRAIN EMULATION",
                  style='Title.TLabel').pack(side='left')

        stats_frame = ttk.Frame(header, style='Dark.TFrame')
        stats_frame.pack(side='right')
        ttk.Label(stats_frame, text=f"Neurons: {self.n_neurons:,}  |  "
                  f"Spikes: {len(self.df):,}  |  "
                  f"Duration: {self.t_max:.0f} ms",
                  style='Stat.TLabel').pack()

        # Main figure area
        fig_frame = ttk.Frame(self.root, style='Dark.TFrame')
        fig_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.fig = plt.Figure(figsize=(16, 9), facecolor='#0a0a0a')
        self.fig.subplots_adjust(hspace=0.35, left=0.06, right=0.97, top=0.95, bottom=0.08)

        # Subplot 1: Spike Raster
        self.ax_raster = self.fig.add_subplot(3, 1, 1)
        self._style_axis(self.ax_raster, "Spike Raster Plot")

        # Subplot 2: Activity Heatmap
        self.ax_heatmap = self.fig.add_subplot(3, 1, 2)
        self._style_axis(self.ax_heatmap, "Neural Activity Heatmap")

        # Subplot 3: Population firing rate
        self.ax_rate = self.fig.add_subplot(3, 1, 3)
        self._style_axis(self.ax_rate, "Population Firing Rate")

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Controls
        controls = ttk.Frame(self.root, style='Dark.TFrame')
        controls.pack(fill='x', padx=10, pady=(0, 10))

        self.play_btn = tk.Button(controls, text="PLAY", command=self._toggle_play,
                                   bg='#1b4f8a', fg='white', font=('Consolas', 11, 'bold'),
                                   width=8, relief='flat', activebackground='#00b4d8')
        self.play_btn.pack(side='left', padx=5)

        tk.Button(controls, text="RESET", command=self._reset,
                  bg='#333', fg='white', font=('Consolas', 11),
                  width=8, relief='flat', activebackground='#555').pack(side='left', padx=5)

        ttk.Label(controls, text="Time:", style='Dark.TLabel').pack(side='left', padx=(20, 5))

        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = tk.Scale(controls, from_=0, to=self.n_bins - 1,
                                     orient='horizontal', variable=self.time_var,
                                     command=self._on_slider, showvalue=False,
                                     bg='#0a0a0a', fg='#48cae4', troughcolor='#1b4f8a',
                                     highlightthickness=0, length=500)
        self.time_slider.pack(side='left', fill='x', expand=True, padx=5)

        self.time_label = ttk.Label(controls, text="0.0 ms", style='Stat.TLabel')
        self.time_label.pack(side='left', padx=10)

        self.spike_label = ttk.Label(controls, text="Active: 0", style='Stat.TLabel')
        self.spike_label.pack(side='left', padx=10)

        # Draw initial static plots
        self._draw_static()
        self._update_frame(0)

    def _style_axis(self, ax, title):
        ax.set_facecolor('#0a0a0a')
        ax.set_title(title, color='#48cae4', fontsize=11, fontfamily='monospace', pad=8)
        ax.tick_params(colors='#666', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333')

    def _draw_static(self):
        ax = self.ax_raster
        ax.clear()
        self._style_axis(ax, "Spike Raster Plot")

        for nid in self.neurons:
            y = self.neuron_idx[nid]
            times = self.spike_times[nid]
            ax.scatter(times, np.full_like(times, y), s=0.3, c='#00b4d8', alpha=0.6, linewidths=0)

        ax.set_xlim(0, self.t_max)
        ax.set_ylim(-1, self.n_neurons)
        ax.set_ylabel('Neuron #', color='#888', fontsize=9)
        self.raster_line = ax.axvline(x=0, color='#ff4444', linewidth=1, alpha=0.8)

        # Heatmap
        ax2 = self.ax_heatmap
        ax2.clear()
        self._style_axis(ax2, "Neural Activity Heatmap (sorted by spike count)")

        # Show top 80 most active neurons for clarity
        n_show = min(80, self.n_neurons)
        display_data = self.activity_sorted[:n_show]
        vmax = max(display_data.max(), 1)

        self.heatmap_img = ax2.imshow(
            display_data, aspect='auto', cmap=self.cmap,
            extent=[0, self.t_max, n_show, 0],
            vmin=0, vmax=vmax, interpolation='nearest'
        )
        ax2.set_ylabel('Neuron rank', color='#888', fontsize=9)
        self.heatmap_line = ax2.axvline(x=0, color='#ff4444', linewidth=1, alpha=0.8)

        # Population rate
        ax3 = self.ax_rate
        ax3.clear()
        self._style_axis(ax3, "Population Firing Rate (spikes / 5ms bin)")

        pop_rate = self.activity_matrix.sum(axis=0)
        bin_centers = (self.time_bins[:-1] + self.time_bins[1:]) / 2

        ax3.fill_between(bin_centers, pop_rate, alpha=0.3, color='#00b4d8')
        ax3.plot(bin_centers, pop_rate, color='#48cae4', linewidth=0.8)
        ax3.set_xlim(0, self.t_max)
        ax3.set_ylim(0, max(pop_rate.max() * 1.1, 1))
        ax3.set_xlabel('Time (ms)', color='#888', fontsize=9)
        ax3.set_ylabel('Spikes', color='#888', fontsize=9)
        self.rate_line = ax3.axvline(x=0, color='#ff4444', linewidth=1, alpha=0.8)

        self.canvas.draw()

    def _update_frame(self, bin_idx):
        bin_idx = int(bin_idx)
        t_ms = self.time_bins[bin_idx]

        self.raster_line.set_xdata([t_ms, t_ms])
        self.heatmap_line.set_xdata([t_ms, t_ms])
        self.rate_line.set_xdata([t_ms, t_ms])

        self.time_label.config(text=f"{t_ms:.0f} ms")

        # Count active neurons in current bin
        if bin_idx < self.n_bins:
            active = int((self.activity_matrix[:, bin_idx] > 0).sum())
            spikes_now = int(self.activity_matrix[:, bin_idx].sum())
        else:
            active = 0
            spikes_now = 0
        self.spike_label.config(text=f"Active: {active} | Spikes: {spikes_now}")

        self.canvas.draw_idle()

    def _on_slider(self, val):
        self._update_frame(float(val))

    def _toggle_play(self):
        if self.playing:
            self.playing = False
            self.play_btn.config(text="PLAY", bg='#1b4f8a')
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
        else:
            self.playing = True
            self.play_btn.config(text="PAUSE", bg='#ff4444')
            self._animate()

    def _animate(self):
        if not self.playing:
            return
        self.current_bin = int(self.time_var.get()) + 1
        if self.current_bin >= self.n_bins:
            self.current_bin = 0
        self.time_var.set(self.current_bin)
        self._update_frame(self.current_bin)
        self.anim_id = self.root.after(self.speed, self._animate)

    def _reset(self):
        self.playing = False
        self.play_btn.config(text="PLAY", bg='#1b4f8a')
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
        self.current_bin = 0
        self.time_var.set(0)
        self._update_frame(0)


def main():
    parquet_path = Path(__file__).parent / 'data' / 'results' / 'pytorch_t1.0s_n1.parquet'

    if not parquet_path.exists():
        print(f"No simulation data found at {parquet_path}")
        print("Run the simulation first: python main.py --pytorch --t_run 1 --n_run 1")
        return

    print("Loading spike data...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} spikes from {df['flywire_id'].nunique()} neurons")
    print("Launching visualizer...")

    root = tk.Tk()
    app = FlyBrainVisualizer(root, df)
    root.mainloop()


if __name__ == '__main__':
    main()
