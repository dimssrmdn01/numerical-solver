import streamlit as st
import pandas as pd
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Numerical Methods Solver", layout="wide")
st.title("🧮 Interactive Numerical Methods Solver (Pure Numerical Edition)")
st.markdown("A computational tool for iterative root-finding algorithms. Designed for academic and analytical use.")
st.divider()

# --- MATHEMATICAL ENGINE ---
class NumericalSolver:
    def __init__(self, equation_str):
        self.equation_str = equation_str
        
    def evaluate_function(self, x_val):
        """Evaluates the mathematical string input using numpy."""
        allowed_names = {k: v for k, v in np.__dict__.items() if not k.startswith("__")}
        allowed_names['x'] = x_val
        try:
            return eval(self.equation_str, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            raise ValueError(f"Function evaluation error: {e}")
            
    def evaluate_derivative_numerical(self, x_val, h=1e-5):
        """Calculates f'(x) numerically using the forward difference method."""
        try:
            fx_plus_h = self.evaluate_function(x_val + h)
            fx = self.evaluate_function(x_val)
            return (fx_plus_h - fx) / h
        except Exception as e:
            raise ValueError(f"Numerical derivative error: {e}")

    def bisection(self, a, b, tol, max_iter):
        if self.evaluate_function(a) * self.evaluate_function(b) >= 0:
            return None, "Initial Condition Error: f(a) and f(b) must have opposite signs."
        
        data = []
        for i in range(1, int(max_iter) + 1):
            c = (a + b) / 2.0
            fa = self.evaluate_function(a)
            fc = self.evaluate_function(c)
            error = abs(b - a)
            
            data.append({"Iteration": i, "a": a, "b": b, "c (Midpoint)": c, "f(a)": fa, "f(c)": fc, "Error": error})
            
            if abs(fc) < tol or error < tol: break
            if fa * fc < 0: b = c 
            else: a = c 
        return pd.DataFrame(data), None, "c (Midpoint)"

    def regula_falsi(self, a, b, tol, max_iter):
        if self.evaluate_function(a) * self.evaluate_function(b) >= 0:
            return None, "Initial Condition Error: f(a) and f(b) must have opposite signs."
        
        data = []
        for i in range(1, int(max_iter) + 1):
            fa = self.evaluate_function(a)
            fb = self.evaluate_function(b)
            if fb - fa == 0: return None, "Division by zero."
            
            c = (a * fb - b * fa) / (fb - fa)
            fc = self.evaluate_function(c)
            error = abs(fc)
            
            data.append({"Iteration": i, "a": a, "b": b, "c (Root Estimate)": c, "f(a)": fa, "f(b)": fb, "f(c)": fc, "Error": error})
            
            if abs(fc) < tol: break
            if fa * fc < 0: b = c 
            else: a = c 
        return pd.DataFrame(data), None, "c (Root Estimate)"

    def newton_raphson(self, x0, tol, max_iter):
        data = []
        x_curr = x0
        for i in range(1, int(max_iter) + 1):
            fx = self.evaluate_function(x_curr)
            dfx = self.evaluate_derivative_numerical(x_curr) # Panggilan metode numerik baru
            
            if dfx == 0:
                return None, "Mathematical Error: Derivative is zero (division by zero)."
                
            x_next = x_curr - (fx / dfx)
            error = abs(x_next - x_curr)
            
            data.append({
                "Iteration": i, 
                "x_i": x_curr, 
                "f(x_i)": fx, 
                "f'(x_i)": dfx, 
                "x_{i+1}": x_next, 
                "Error": error
            })
            
            if error < tol or abs(self.evaluate_function(x_next)) < tol:
                break
            x_curr = x_next
            
        return pd.DataFrame(data), None, "x_{i+1}"

# --- SIDEBAR: PARAMETER CONFIGURATION ---
st.sidebar.header("Algorithm Parameters")
method_choice = st.sidebar.selectbox("Select Numerical Method", ["Bisection Method", "Regula Falsi Method", "Newton-Raphson Method"])
equation_input = st.sidebar.text_input("Function f(x)", value="x**3 - x - 2")

if method_choice == "Newton-Raphson Method":
    st.sidebar.markdown("Requires only one initial guess ($x_0$).")
    x0_input = st.sidebar.number_input("Initial Guess (x0)", value=1.5)
else:
    st.sidebar.markdown("Requires an interval $[a, b]$ where signs change.")
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
        
        if method_choice == "Newton-Raphson Method":
            st.markdown(f"**Derivative Method:** `Approximated via Finite Differences (h = 1e-5)`")
        
        with st.spinner(f"Executing {method_choice}..."):
            if method_choice == "Bisection Method":
                df_result, error_msg, root_col = solver.bisection(a_input, b_input, tol_input, max_iter_input)
            elif method_choice == "Regula Falsi Method":
                df_result, error_msg, root_col = solver.regula_falsi(a_input, b_input, tol_input, max_iter_input)
            else:
                df_result, error_msg, root_col = solver.newton_raphson(x0_input, tol_input, max_iter_input)
            
            if error_msg:
                st.error(error_msg)
            else:
                final_root = df_result[root_col].iloc[-1]
                total_iterations = len(df_result)
                
                st.success(f"Convergence achieved at iteration {total_iterations}. Estimated Root (x): **{final_root:.6f}**")
                
                st.subheader("Iteration History Matrix")
                st.dataframe(df_result, use_container_width=True)
                
                st.subheader("Error Convergence Profile")
                st.line_chart(df_result.set_index("Iteration")["Error"])
                
    except Exception as e:
        st.error(f"Syntax or Execution Error: {e}")