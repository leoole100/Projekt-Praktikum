#%%
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("../style.mplstyle")
from model import Model

# %%

# Physical constants
h = 6.626e-34   # Planck constant (J·s)
c = 300e6       # Speed of light (m/s)
kB = 1.381e-23  # Boltzmann constant (J/K)

def B(l, T): 
    """Spectral radiance (Planck-like) as function of wavelength and temperature."""
    return 2*h*c**2 / l**5 / (np.exp(h * c / (l * kB * T)) - 1)

def norm(a): 
    """Normalize array to its maximum."""
    return a / a.max()

def plot_streak(model:Model, l = np.linspace(100, 2000, 100) * 1e-9):
    """
    Create a streak plot from a Model instance.
    
    Returns:
        fig (matplotlib.figure.Figure): The generated figure object.
    """
    if model.t is None or model.T_e is None: m()

    b = B(l[:, None], model.T_e[None, :])  # shape: (λ, t)
    t = model.t * 1e15                     # time (fs)

    fig, ax = plt.subplots(2, 2, gridspec_kw={
        'height_ratios': [1, 3],
        'width_ratios': [3, 1]
    })

    # Top left: Temperature vs time
    ax[0, 0].plot(t, model.T_e / 1e3)
    ax[0, 0].set_ylabel(r"$T_e$ (kK)")

    # Bottom left: Spectrogram
    ax[1, 0].contourf(t, l / 1e-9, b, vmin=0)
    ax[1, 0].set_ylabel(r"$\lambda$ (nm)")
    ax[1, 0].set_xlabel("t (fs)")

    # Bottom right: Time-averaged spectrum
    ax[1, 1].plot(norm(b.mean(axis=1)), l, label="mean")
    ax[1, 1].set_xlabel("summed")
    ax[1, 1].set_ylim(l.min(), l.max())

    # Top right: blank
    ax[0, 1].axis("off")

    for a in ax.flatten():
        a.label_outer()

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    m = Model()
    m.fwhm = 10e-15
    m.t = np.linspace(-3 * m.sigma, .5 * (m.sigma + m.g), 1000)
    m()
    fig = plot_streak(m)
    fig.savefig("figures/streak view.pdf")
    plt.show()

# %%
