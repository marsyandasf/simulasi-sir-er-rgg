"""
Simulasi SIR stokastik pada graf Erdős–Rényi (ER) vs Random Geometric Graph (RGG)
dengan ilustrasi parameter penyakit mosaik singkong (CMD).

Simpul = petak lahan, sisi = kemungkinan penularan antarpetak (lewat kutu kebul).
Status simpul: S (rentan), I (terinfeksi), R (removed = dicabut/dipanen).

Satu langkah waktu = 1 minggu. Tiap langkah:
  1. Setiap petak I mencoba menulari setiap tetangga S dengan peluang beta.
  2. Setiap petak I (termasuk yang tertular di langkah sebelumnya, bukan langkah ini)
     berpindah ke R dengan peluang gamma.
Dengan urutan ini, transmisibilitas per sisi adalah
     T = beta / (beta + gamma - beta*gamma)
sehingga ambang epidemik ER (distribusi derajat Poisson) adalah T * <k> = 1.
"""

import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# 1. PARAMETER BIOLOGIS (sumber: Jittamai et al. 2021, Math. Biosci. Eng.)
# ---------------------------------------------------------------------------
DT_HARI = 7                 # 1 langkah = 1 minggu
ROGUING_HARI = 0.03         # laju cabut tanaman sakit, baseline (rentang 0.03-0.1 /hari)
PANEN_HARI = 0.003          # laju panen (/hari)


def laju_ke_peluang(laju_per_hari, dt=DT_HARI):
    """Konversi laju kontinu (/hari) menjadi peluang per langkah: p = 1 - exp(-laju*dt)."""
    return 1.0 - math.exp(-laju_per_hari * dt)


def gamma_dari_roguing(roguing_hari, panen_hari=PANEN_HARI, dt=DT_HARI):
    """Peluang removed per langkah = cabut + panen (dua laju kompetitif dijumlahkan)."""
    return laju_ke_peluang(roguing_hari + panen_hari, dt)


# ---------------------------------------------------------------------------
# 2. PEMBANGKIT GRAF
# ---------------------------------------------------------------------------
def graf_er(N, k_rata, rng):
    """G(N,p) dengan p = <k>/(N-1). Mengembalikan matriks adjacency CSR."""
    p = k_rata / (N - 1)
    G = nx.fast_gnp_random_graph(N, p, seed=int(rng.integers(2**31)))
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.int8), None


def graf_rgg(N, k_rata, rng, torus=True):
    """
    RGG: N titik acak seragam di [0,1]^2, dua titik terhubung jika jaraknya < r.
    Dengan batas periodik (torus), <k> = (N-1)*pi*r^2 sehingga r = sqrt(<k>/(pi*(N-1))).
    Mengembalikan (adjacency CSR, posisi titik).
    """
    r = math.sqrt(k_rata / (math.pi * (N - 1)))
    pos = rng.random((N, 2))
    tree = cKDTree(pos, boxsize=1.0 if torus else None)
    pasangan = tree.query_pairs(r, output_type="ndarray")
    i, j = pasangan[:, 0], pasangan[:, 1]
    data = np.ones(2 * len(i), dtype=np.int8)
    A = sp.csr_array((data, (np.r_[i, j], np.r_[j, i])), shape=(N, N))
    return A, pos


PEMBANGKIT = {"ER": graf_er, "RGG": graf_rgg}


# ---------------------------------------------------------------------------
# 3. SIMULASI SIR STOKASTIK WAKTU DISKRIT
# ---------------------------------------------------------------------------
S, I, R = 0, 1, 2


def simulasi_sir(A, beta, gamma, rng, n_awal=None, frac_awal=0.01, t_maks=2000,
                 simpan_status_akhir=False):
    """
    Satu realisasi SIR pada graf dengan adjacency CSR A.
    Mengembalikan dict berisi deret waktu S(t), I(t), R(t) (dalam proporsi).
    """
    N = A.shape[0]
    indptr, indices = A.indptr, A.indices
    status = np.zeros(N, dtype=np.int8)
    n0 = n_awal if n_awal is not None else max(1, int(round(frac_awal * N)))
    status[rng.choice(N, n0, replace=False)] = I

    riwayat = [(N - n0, n0, 0)]
    for _ in range(t_maks):
        terinfeksi = np.flatnonzero(status == I)
        if terinfeksi.size == 0:
            break
        # --- transmisi: kumpulkan semua tetangga dari simpul terinfeksi ---
        awal, akhir = indptr[terinfeksi], indptr[terinfeksi + 1]
        panjang = akhir - awal
        if panjang.sum() > 0:
            idx = np.repeat(awal - np.cumsum(np.r_[0, panjang[:-1]]), panjang) + np.arange(panjang.sum())
            tetangga = indices[idx]
            # m_j = banyaknya tetangga terinfeksi dari simpul j
            m = np.bincount(tetangga, minlength=N)
            kandidat = np.flatnonzero((m > 0) & (status == S))
            # peluang tertular = 1 - (1-beta)^m  (percobaan independen per sisi)
            p_tular = 1.0 - (1.0 - beta) ** m[kandidat]
            baru = kandidat[rng.random(kandidat.size) < p_tular]
        else:
            baru = np.empty(0, dtype=int)
        # --- removal: hanya simpul yang sudah I sebelum langkah ini ---
        sembuh = terinfeksi[rng.random(terinfeksi.size) < gamma]
        status[sembuh] = R
        status[baru] = I
        riwayat.append(np.bincount(status, minlength=3))

    riwayat = np.asarray(riwayat, dtype=float) / N
    hasil = {"S": riwayat[:, 0], "I": riwayat[:, 1], "R": riwayat[:, 2]}
    hasil["ukuran_akhir"] = hasil["R"][-1] + hasil["I"][-1]
    hasil["puncak_I"] = hasil["I"].max()
    hasil["waktu_puncak"] = int(hasil["I"].argmax())
    hasil["durasi"] = len(hasil["I"]) - 1
    if simpan_status_akhir:
        hasil["status"] = status
    return hasil


# ---------------------------------------------------------------------------
# 4. TEORI ER (pembanding analitik)
# ---------------------------------------------------------------------------
def transmisibilitas(beta, gamma):
    return beta / (beta + gamma - beta * gamma)


def beta_kritis_er(k_rata, gamma):
    """Solusi T*<k> = 1  ->  beta_c = gamma / (<k> - 1 + gamma)."""
    return gamma / (k_rata - 1 + gamma)


def ukuran_akhir_teori_er(beta, gamma, k_rata, iterasi=500):
    """
    Ukuran wabah besar di ER (limit N besar) lewat perkolasi ikatan:
    s = 1 - exp(-T <k> s). Diselesaikan dengan iterasi titik tetap.
    """
    Tk = transmisibilitas(beta, gamma) * k_rata
    if Tk <= 1:
        return 0.0
    s = 0.5
    for _ in range(iterasi):
        s = 1.0 - math.exp(-Tk * s)
    return s


# ---------------------------------------------------------------------------
# 5. MONTE CARLO
# ---------------------------------------------------------------------------
def monte_carlo(nama_graf, N, k_rata, beta, gamma, M, rng, graf_baru_tiap_ulangan=True,
                simpan_kurva=False, **kw):
    """
    Ulangi simulasi M kali. Graf dibangkitkan ulang tiap ulangan (rata-rata atas
    ensembel graf, bukan hanya satu graf). Mengembalikan ringkasan statistik.
    """
    buat = PEMBANGKIT[nama_graf]
    A, _ = buat(N, k_rata, rng)
    ukuran, puncak, waktu_puncak, kurva = [], [], [], []
    for m in range(M):
        if graf_baru_tiap_ulangan and m > 0:
            A, _ = buat(N, k_rata, rng)
        h = simulasi_sir(A, beta, gamma, rng, **kw)
        ukuran.append(h["ukuran_akhir"])
        puncak.append(h["puncak_I"])
        waktu_puncak.append(h["waktu_puncak"])
        if simpan_kurva:
            kurva.append(h["I"])
    ukuran = np.asarray(ukuran)
    ringkas = {
        "graf": nama_graf, "N": N, "k": k_rata, "beta": beta, "gamma": gamma,
        "T": transmisibilitas(beta, gamma),
        "ukuran_akhir_rata": ukuran.mean(),
        "ukuran_akhir_sd": ukuran.std(ddof=1),
        "peluang_wabah_besar": (ukuran > 0.10).mean(),   # definisi 'wabah besar' > 10% petak
        "puncak_I_rata": np.mean(puncak),
        "waktu_puncak_rata": np.mean(waktu_puncak),
    }
    if simpan_kurva:
        ringkas["kurva_I"] = kurva
    return ringkas


def estimasi_beta_kritis(beta_grid, nilai, batas=0.10):
    """
    Estimasi numerik ambang: beta terkecil di mana rata-rata ukuran akhir melewati `batas`,
    diinterpolasi linear pada skala log beta. Mengembalikan nan jika tidak pernah melewati.
    """
    beta_grid, nilai = np.asarray(beta_grid), np.asarray(nilai)
    atas = np.flatnonzero(nilai >= batas)
    if atas.size == 0 or atas[0] == 0:
        return float("nan")
    j = atas[0]
    x0, x1 = math.log(beta_grid[j - 1]), math.log(beta_grid[j])
    y0, y1 = nilai[j - 1], nilai[j]
    return math.exp(x0 + (batas - y0) * (x1 - x0) / (y1 - y0))
