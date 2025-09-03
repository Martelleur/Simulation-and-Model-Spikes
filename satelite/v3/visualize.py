from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    out = Path("out")
    df = pd.read_csv(out / "states.csv")

    # Plot 1: Altitude over time
    plt.figure()
    plt.plot(df["t"], df["alt"])
    plt.title("Altitude over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.savefig(out / "plot_altitude.png", bbox_inches="tight")
    plt.show()

    # Plot 2: Simple ground-track proxy
    # Using atan2(ry, rx) as a crude longitude angle (not Earth-rotating), purely illustrative.
    lon = np.arctan2(df["ry"].values, df["rx"].values)
    plt.figure()
    plt.plot(df["t"], lon)
    plt.title("Ground-track longitude proxy (ECI atan2) over time")
    plt.xlabel("Time (s)")
    plt.ylabel("atan2(ry,rx) [rad]")
    plt.savefig(out / "plot_groundtrack_proxy.png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
