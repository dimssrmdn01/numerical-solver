import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Numerical Methods Solver", layout="wide")
st.title("Interactive Numerical Methods Solver")
st.markdown("A computational tool for iterative root-finding algorithms. Designed for academic and analytical use.")
st.divider()

# --- MATHEMATICAL ENGINE ---
class NumericalSolver:
    def __init__(self, equation_str):
        self.equation_str = equation_str
        
    def evaluate_function(self, x_val):
        """Safely evaluates the mathematical string input using numpy."""
        allowed_names = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
        allowed_names['x'] = x_val
        try:
            return eval(self.equation_str, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Function evaluation error: {e}")

    def bisection(self, a, b, tol, max_iter):
        """Executes the Bisection Method."""
        if self.evaluate_function(a) * self.evaluate_function(b) >= 0:
            return None, "Initial Condition Error: f(a) and f(b) must have opposite signs."
        
        data = []
        for i in range(1, int(max_iter) + 1):
            c = (a + b) / 2.0
            fa = self.evaluate_function(a)
            fb = self.evaluate_function(b)
            fc = self.evaluate_function(c)
            
            error = abs(b - a)
            
            data.append({
                "Iteration": i,
                "a": a,
                "b": b,
                "c (Midpoint)": c,
                "f(a)": fa,
                "f(b)": fb,
                "f(c)": fc,
                "Error |b-a|": error
            })
            
            if abs(fc) < tol or error < tol:
                break
                
            if fa * fc < 0:
                b = c 
            else:
                a = c 
                
        return pd.DataFrame(data), None

    def regula_falsi(self, a, b, tol, max_iter):
        """Executes the Regula Falsi (False Position) Method."""
        if self.evaluate_function(a) * self.evaluate_function(b) >= 0:
            return None, "Initial Condition Error: f(a) and f(b) must have opposite signs."
        
        data = []
        for i in range(1, int(max_iter) + 1):
            fa = self.evaluate_function(a)
            fb = self.evaluate_function(b)
            
            if fb - fa == 0:
                return None, "Mathematical Error: Division by zero encountered in Regula Falsi formula."
                
            # Formula linear perkalian silang Regula Falsi
            c = (a * fb - b * fa) / (fb - fa)
            fc = self.evaluate_function(c)
            
            error = abs(fc)
            
            data.append({
                "Iteration": i,
                "a": a,
                "b": b,
                "c (Root Estimate)": c,
                "f(a)": fa,
                "f(b)": fb,
                "f(c)": fc,
                "Error |f(c)|": error
            })
            
            if abs(fc) < tol:
                break
                
            if fa * fc < 0:
                b = c 
            else:
                a = c 
                
        return pd.DataFrame(data), None

# --- SIDEBAR: PARAMETER CONFIGURATION ---
st.sidebar.header("Algorithm Parameters")

# Menu Dropdown untuk memilih metode
method_choice = st.sidebar.selectbox("Select Numerical Method", ["Bisection Method", "Regula Falsi Method"])

equation_input = st.sidebar.text_input("Function f(x)", value="x**3 - x - 2")

col_param1, col_param2 = st.sidebar.columns(2)
a_input = col_param1.number_input("Lower Bound (a)", value=1.0)
b_input = col_param2.number_input("Upper Bound (b)", value=2.0)

col_param3, col_param4 = st.sidebar.columns(2)
tol_input = col_param3.number_input("Tolerance", value=0.001, format="%.5f")
max_iter_input = col_param4.number_input("Max Iterations", value=50, step=1)

# --- EXECUTION & UI RENDERING ---
if st.sidebar.button("Compute Root"):
    try:
        solver = NumericalSolver(equation_input)
        st.markdown(f"**Target Function:** `f(x) = {equation_input}`")
        
        with st.spinner(f"Executing {method_choice}..."):
            # Percabangan eksekusi berdasarkan pilihan pengguna
            if method_choice == "Bisection Method":
                df_result, error_msg = solver.bisection(a_input, b_input, tol_input, max_iter_input)
                error_col = "Error |b-a|"
                root_col = "c (Midpoint)"
            else:
                df_result, error_msg = solver.regula_falsi(a_input, b_input, tol_input, max_iter_input)
                error_col = "Error |f(c)|"
                root_col = "c (Root Estimate)"
            
            if error_msg:
                st.error(error_msg)
            else:
                final_root = df_result[root_col].iloc[-1]
                total_iterations = len(df_result)
                
                # Menampilkan Ringkasan Hasil
                st.success(f"Convergence achieved at iteration {total_iterations}. Estimated Root (x): **{final_root:.6f}**")
                
                # Menampilkan Tabel Matriks Riwayat Iterasi
                st.subheader("Iteration History Matrix")
                st.dataframe(df_result, use_container_width=True)
                
                # Menampilkan Grafik Konvergensi Error
                st.subheader("Error Convergence Profile")
                st.line_chart(df_result.set_index("Iteration")[error_col])
                
    except Exception as e:
        st.error(f"Syntax or Execution Error: {e}")