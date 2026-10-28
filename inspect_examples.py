

import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

import config
import catalog
from lightcurves import get_clean_lightcurve
from views import transit_mask, flatten_for_transit, make_global_view, make_local_view

N_EXAMPLES_PER_CLASS = 5


def pick_examples(df, n_per_class=N_EXAMPLES_PER_CLASS):
    
    confirmed = df[df["koi_disposition"] == "CONFIRMED"].sample(
        n=n_per_class, random_state=config.RANDOM_SEED
    )
    false_pos = df[df["koi_disposition"] == "FALSE POSITIVE"].sample(
        n=n_per_class, random_state=config.RANDOM_SEED
    )
    return confirmed, false_pos


def inspect_one(row, label_name, ax_raw, ax_flat, ax_folded):
    
    kepid = int(row["kepid"])
    period = float(row["koi_period"])
    t0 = float(row["koi_time0bk"])
    duration_hours = float(row["koi_duration"])

    lc = get_clean_lightcurve(kepid)
    if lc is None:
        for ax in (ax_raw, ax_flat, ax_folded):
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return

    
    ax_raw.scatter(lc.time.value, lc.flux.value, s=1, color="black")
    ax_raw.set_title(f"{label_name} kepid={kepid}\nraw (Stage A only)")

    
    mask = transit_mask(lc.time.value, period, t0, duration_hours)
    flat_lc = flatten_for_transit(lc, period, t0, duration_hours)
    ax_flat.scatter(flat_lc.time.value, flat_lc.flux.value, s=1, color="darkblue")
    
    ax_flat.scatter(
        flat_lc.time.value[mask], flat_lc.flux.value[mask],
        s=3, color="red", label="protected (transit mask)"
    )
    ax_flat.legend(fontsize=6)
    ax_flat.set_title("flattened (transit protected)")

    
    global_view = make_global_view(flat_lc, period, t0)
    local_view = make_local_view(flat_lc, period, t0, duration_hours)
    ax_folded.plot(np.linspace(-0.5, 0.5, len(global_view)), global_view,
                   color="gray", alpha=0.5, label="global")
    ax_2 = ax_folded.twinx()
    ax_2.plot(np.linspace(-0.5, 0.5, len(local_view)) * 0.05, local_view,
              color="green", label="local (zoomed)")
    ax_folded.set_title("folded: global (gray) vs local zoom (green)")


def main():
    df = catalog.load_or_fetch_catalog()
    confirmed, false_pos = pick_examples(df)

    n_total = len(confirmed) + len(false_pos)
    fig, axes = plt.subplots(n_total, 3, figsize=(15, 3 * n_total))

    row_idx = 0
    for _, row in confirmed.iterrows():
        inspect_one(row, "CONFIRMED", *axes[row_idx])
        row_idx += 1
    for _, row in false_pos.iterrows():
        inspect_one(row, "FALSE POSITIVE", *axes[row_idx])
        row_idx += 1

    plt.tight_layout()
    out_path = config.DATA_DIR / "phase3_inspection.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved inspection grid to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()