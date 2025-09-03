from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
from sim.producer import run_producer, SimParams
from sim.consumer import run_consumer

def main():
    outdir = Path("out")
    outdir.mkdir(exist_ok=True)
    q = mp.Queue(maxsize=256)

    # Example: ~500 km circular-ish LEO (a ≈ Re + 500 km), e ~ 0.001
    a = 6378.137e3 + 500e3
    params = SimParams(
        a=a, e=0.001, inc=0.0,
        dt=0.5,            # integrator step (s)
        t_total=5400.0,    # simulate 1.5 hours
        sample_state_hz=2, # 2 Hz state samples
        sample_frame_hz=0.5 # 0.5 Hz frames
    )

    p_cons = mp.Process(target=run_consumer, args=(q, str(outdir)))
    p_prod = mp.Process(target=run_producer, args=(q, params))
    p_cons.start()
    p_prod.start()
    p_prod.join()
    p_cons.join()

    print(f"Done. Outputs in: {outdir.resolve()}")
    print(" - states.csv (time, position, velocity, altitude)")
    print(" - frames/frame_*.png (synthetic sensor frames)")
    print(" - summary.json / summary_extra.json")

if __name__ == "__main__":
    main()
