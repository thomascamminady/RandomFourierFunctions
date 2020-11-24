# %% codecell
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import streamlit as st


@njit
def fourier(x, coeffs):
    a = coeffs[0:-1:2]
    b = coeffs[1:-1:2]
    y = a[0]/2*np.ones_like(x)
    for i, (ai, bi) in enumerate(zip(a, b)):
        y += ai*np.cos(2*np.pi*(i+1)*x) + bi*np.sin(2*np.pi*(i+1)*x)
    return y


@njit
def integral(coeffs):
    a = coeffs[0:-1:2]
    b = coeffs[1:-1:2]
    y = a[0] / 2
    for i, (ai, bi) in enumerate(zip(a, b)):
        k = np.pi*(i+1)
        y += np.sin(k)/k * (ai*np.cos(k)+bi*np.sin(k))
    return y


st.title("Random Fourier Functions")
st.markdown("This application creates random functions of the following type:")
st.latex(
    r'''f_{coeffs}(x) = a_0/2 +
    \sum_{i=1}^N {a_i}\, \cos(2\pi \, i \, x)+b_i\, \sin(2\pi \, i \, x)''')
st.markdown("The coefficients $a_i$ and $b_i$ are of order $O(i^{-smoothness})$. ")
st.markdown("More precisely, $a_i = r_i \cdot i^{-smoothness}$ for a random number $r_i$.")

smoothness = st.sidebar.slider('Smoothness', 0.0, 10.0, 2.0, 0.1)
npoints = st.sidebar.slider('Number of points in plot', 100, 10000, 1000, 100)
fixorder = st.sidebar.checkbox('Fix expansion order?', 10)
fixordervalue = st.sidebar.slider('Expansion order value', 1, 1000, 100, 1)
nsamples = st.sidebar.slider("Number of samples", 1, 200, 3, 1)
sampling = st.sidebar.selectbox(
    'Which random nunber generator?',
    ('rand', 'randn'))
fixrandomseed = st.sidebar.checkbox('Fix random seed?', True)
randomseed = st.sidebar.text_input("Random seed value", 1)
normalizezero = st.sidebar.checkbox('Fix integral to be zero?', True)

if fixorder:
    order = fixordervalue
else:
    order = int((10**-8)**(-1/smoothness))

if fixrandomseed:
    rng = np.random.RandomState(int(randomseed))
else:
    rng = np.random.RandomState(np.random.randint(2**32-1))

n = 2*order+1
x = np.linspace(0, 1, npoints)

plt.style.use("https://raw.githubusercontent.com/camminady/kitstyle/master/kitishnotex.mplstyle")
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
for _ in range(nsamples):
    if sampling == "rand":
        coeffs = rng.uniform(-1, 1, n)
    else:
        coeffs = rng.normal(0, 1, n)
    for i in range(1, n, 2):
        coeffs[i] *= i**(-smoothness)
        coeffs[i+1] *= i**(-smoothness)
    if normalizezero:
        y = integral(coeffs)
        coeffs[0] -= 2*y  # integral now has value 0
    axs[0].plot(x, fourier(x, coeffs), lw=1, alpha=0.7)
    if order > 1:
        axs[1].loglog(np.arange(order), np.abs(coeffs[0:-1:2]), '.', markersize=3, alpha=0.7)

axs[0].set_ylim([-2, 2])
axs[0].set_xlabel("$x$")
axs[0].set_ylabel("$f_{coeffs}(x)$", rotation=90, labelpad=0)
axs[1].set_ylim([10**-8, 10**1])
axs[1].set_ylabel("$|coeffs|$", rotation=90, labelpad=0)
axs[1].set_xlabel("$n$")


st.pyplot(fig)
