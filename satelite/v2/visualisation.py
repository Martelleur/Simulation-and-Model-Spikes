import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

def plot_distance_vs_time(df):
    plt.figure()
    plt.plot(df["time_s"]/60.0, df["distance_km"])
    plt.xlabel("Time (minutes)")
    plt.ylabel("Separation distance (km)")
    plt.title("Two-satellite separation over time")
    plt.grid(True)
    plt.show()

def plot_orbits(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df["r1_x_km"], df["r1_y_km"], df["r1_z_km"], label="Sat 1")
    ax.plot(df["r2_x_km"], df["r2_y_km"], df["r2_z_km"], label="Sat 2")
    Re = 6378.137
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = Re*np.cos(u)*np.sin(v)
    y = Re*np.sin(u)*np.sin(v)
    z = Re*np.cos(v)
    ax.plot_wireframe(x, y, z, linewidth=0.2)
    ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.set_zlabel("Z (km)")
    ax.set_title("Orbits in ECI (two-body)")
    ax.legend()
    plt.show()
