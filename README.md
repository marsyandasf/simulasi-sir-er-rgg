# Simulasi SIR Stokastik pada Graf Erdős–Rényi vs Random Geometric Graph

Kode pendukung skripsi *Analisis Komparatif Penyebaran Epidemi SIR Stokastik pada Graf Erdős–Rényi dan Graf Geometris Acak: Studi Kasus Parameter Penyakit Mosaik Singkong*.

## Isi

| File | Isi |
|---|---|
| `sir_graf.py` | Modul inti: pembangkit graf ER dan RGG (torus), simulasi SIR waktu diskrit, rumus teori ER, Monte Carlo, estimasi ambang |
| `simulasi_sir_er_vs_rgg.ipynb` | Notebook utama: verifikasi graf, kurva epidemi, ukuran akhir vs β, sweep parameter, tabel ambang, visual spasial |
| `requirements.txt` | Versi library yang dipakai |
| `hasil/` | Gambar dan CSV dari run terakhir |

## Cara menjalankan

**Lokal**
```bash
pip install -r requirements.txt
jupyter notebook simulasi_sir_er_vs_rgg.ipynb
```

**Google Colab**: buka notebook, lalu aktifkan tiga baris `git clone`, `%cd`, dan `pip install` di sel pertama (ganti `USERNAME/NAMA-REPO`).

Sweep parameter (360 kombinasi, M = 100) butuh sekitar 9 menit. Untuk hasil final, ubah `M_SWEEP = 200`.

## Reprodusibilitas

Seed acak dikunci (`SEED = 2026`). Dengan versi library di `requirements.txt`, menjalankan ulang notebook dari atas menghasilkan angka yang sama.

## Parameter utama

- Laju roguing 0 / 0,03 / 0,1 per hari dan laju panen 0,003 per hari (Jittamai et al., 2021), dikonversi ke peluang per minggu: γ = 1 − exp(−(roguing + panen) · 7)
- β disweep 0,001–0,5 (20 nilai), ⟨k⟩ = 4, 8, 12, N = 1000, 1% petak terinfeksi awal
