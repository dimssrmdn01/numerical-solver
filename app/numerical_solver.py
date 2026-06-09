import time
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Tuple, Optional, Callable

# Konfigurasi Halaman Dasar
st.set_page_config(page_title="Numerical Methods Solver", layout="wide")

class DatabaseManager:
    """Mengelola operasi penyimpanan log komputasi ke dalam SQLite database."""
    
    def __init__(self, db_name: str = "solver_audit.db"):
        self.db_name = db_name

    def init_db(self) -> None:
        """Inisialisasi tabel histori komputasi jika belum ada."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS computation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                method TEXT,
                function_str TEXT,
                root_found REAL,
                iterations INTEGER,
                execution_time_ms REAL,
                tolerance REAL
            )
        ''')
        conn.commit()
        conn.close()

    def log_computation(self, method: str, func_str: str, root: float, iters: int, exec_time: float, tol: float) -> None:
        """Merekam hasil pencarian akar persamaan beserta metrik performanya."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO computation_logs (timestamp, method, function_str, root_found, iterations, execution_time_ms, tolerance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, method, func_str, float(root), int(iters), float(exec_time), float(tol)))
        conn.commit()
        conn.close()

    def fetch_logs(self, limit: int = 5) -> pd.DataFrame:
        """Mengambil data log histori komputasi terbaru."""
        conn = sqlite3.connect(self.db_name)
        query = f"SELECT * FROM computation_logs ORDER BY id DESC LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df


class NumericalEngine:
    """Mesin komputasi berorientasi objek untuk metode numerik."""
    
    @staticmethod
    def _evaluate_function(func_str: str, x_val: float) -> float:
        """Evaluasi string fungsi matematika secara aman menggunakan namespace NumPy."""
        allowed_names = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
        allowed_names['x'] = x_val
        try:
            return eval(func_str, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Fungsi tidak valid atau tidak dapat dievaluasi: {e}")

    @classmethod
    def bisection(cls, func_str: str, a: float, b: float, tol: float, max_iter: int) -> Tuple[Optional[float], pd.DataFrame, float]:
        """Implementasi Metode Biseksi dengan pelacakan riwayat iterasi."""
        start_time = time.perf_counter()
        
        fa = cls._evaluate_function(func_str, a)
        fb = cls._evaluate_function(func_str, b)
        
        if fa * fb >= 0:
            raise ValueError("Syarat awal gagal: f(a) dan f(b) harus memiliki tanda yang berlawanan.")
            
        history = []
        root = a
        
        for i in range(max_iter):
            root = (a + b) / 2.0
            f_root = cls._evaluate_function(func_str, root)
            error = abs(f_root)
            
            history.append({
                'Iteration': i + 1,
                'a': a,
                'b': b,
                'Root (x)': root,
                'f(x)': f_root,
                'Error': error
            })
            
            if error < tol:
                break
                
            if cls._evaluate_function(func_str, a) * f_root < 0:
                b = root
            else:
                a = root
                
        exec_time = (time.perf_counter() - start_time) * 1000
        return root, pd.DataFrame(history), exec_time

    @classmethod
    def regula_falsi(cls, func_str: str, a: float, b: float, tol: float, max_iter: int) -> Tuple[Optional[float], pd.DataFrame, float]:
        """Implementasi Metode Regula Falsi dengan pelacakan riwayat iterasi."""
        start_time = time.perf_counter()
        
        fa = cls._evaluate_function(func_str, a)
        fb = cls._evaluate_function(func_str, b)
        
        if fa * fb >= 0:
            raise ValueError("Syarat awal gagal: f(a) dan f(b) harus memiliki tanda yang berlawanan.")
            
        history = []
        root = a
        
        for i in range(max_iter):
            fa = cls._evaluate_function(func_str, a)
            fb = cls._evaluate_function(func_str, b)
            
            root = b - (fb * (b - a)) / (fb - fa)
            f_root = cls._evaluate_function(func_str, root)
            error = abs(f_root)
            
            history.append({
                'Iteration': i + 1,
                'a': a,
                'b': b,
                'Root (x)': root,
                'f(x)': f_root,
                'Error': error
            })
            
            if error < tol:
                break
                
            if fa * f_root < 0:
                b = root
            else:
                a = root
                
        exec_time = (time.perf_counter() - start_time) * 1000
        return root, pd.DataFrame(history), exec_time

# Inisialisasi Database
db = DatabaseManager()
db.init_db()

# Antarmuka Pengguna Streamlit
st.title("Interactive Numerical Methods Solver")
st.markdown("Mesin komputasi web untuk mencari akar persamaan non-linear menggunakan algoritma pencarian iteratif, dilengkapi dengan benchmarking performa dan pencatatan metrik operasional.")
st.write("---")

col_input, col_config = st.columns(2)

with col_input:
    st.subheader("Fungsi & Batasan")
    func_input = st.text_input("Masukkan Fungsi $f(x)$ (Gunakan sintaksis Python/NumPy, contoh: x**3 - x - 2)", value="x**3 - x - 2")
    col_a, col_b = st.columns(2)
    with col_a:
        a_val = st.number_input("Batas Bawah (a)", value=1.0, format="%.4f")
    with col_b:
        b_val = st.number_input("Batas Atas (b)", value=2.0, format="%.4f")

with col_config:
    st.subheader("Konfigurasi Mesin")
    method_select = st.selectbox("Algoritma Pencarian", ["Regula Falsi", "Bisection"])
    tolerance = st.number_input("Batas Toleransi Eror", value=1e-6, format="%.8f")
    max_iterations = st.number_input("Maksimum Iterasi", min_value=1, max_value=1000, value=100)

if st.button("Jalankan Komputasi Numerik", type="primary"):
    try:
        engine = NumericalEngine()
        
        if method_select == "Bisection":
            final_root, df_history, time_ms = engine.bisection(func_input, a_val, b_val, tolerance, max_iterations)
        else:
            final_root, df_history, time_ms = engine.regula_falsi(func_input, a_val, b_val, tolerance, max_iterations)
            
        total_iters = len(df_history)
        
        # Eksekusi logging ke database SQLite
        db.log_computation(method_select, func_input, final_root, total_iters, time_ms, tolerance)
        
        st.success(f"Komputasi berhasil diselesaikan dalam {time_ms:.4f} ms.")
        
        # Panel Metrik Utama
        m1, m2, m3 = st.columns(3)
        m1.metric("Akar Ditemukan (x)", f"{final_root:.6f}")
        m2.metric("Total Siklus Iterasi", f"{total_iters}")
        m3.metric("Waktu Eksekusi", f"{time_ms:.4f} ms")
        
        st.write("---")
        
        col_chart, col_table = st.columns([1.2, 1])
        
        with col_chart:
            st.markdown("### Profil Konvergensi Eror")
            fig = px.line(df_history, x='Iteration', y='Error', markers=True, 
                          title=f"Laju Penurunan Eror Logaritmik - {method_select}",
                          log_y=True)
            st.plotly_chart(fig, use_container_width=True)
            
        with col_table:
            st.markdown("### Matriks Histori Iterasi")
            st.dataframe(df_history, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan komputasi: {e}")

st.write("---")
st.markdown("### Audit Log Histori Komputasi (Database Internal)")
df_logs = db.fetch_logs(limit=5)
if not df_logs.empty:
    st.dataframe(df_logs, use_container_width=True, hide_index=True)
else:
    st.info("Log database belum tersedia. Lakukan komputasi pertama Anda.")