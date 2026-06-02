#  Interactive Numerical Methods Solver

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-150458.svg)

A web-based computational tool designed to solve non-linear equations using iterative numerical algorithms. This application provides step-by-step iteration matrices and error convergence visualizations, making it an excellent companion for academic learning and mathematical analysis.

🔗 **[Live Demo: Numerical Solver App](TULIS_LINK_STREAMLIT_KAMU_DISINI)**

##  Features
* **Multiple Algorithms**: Currently supports **Bisection Method** and **Regula Falsi (False Position) Method**.
* **Dynamic Function Parser**: Input complex mathematical functions securely using standard Python/NumPy syntax (e.g., `x**3 - x - 2` or `np.sin(x)`).
* **Iteration History Matrix**: Generates a detailed, step-by-step tabular breakdown of the root-finding process.
* **Error Convergence Profile**: Visualizes how the algorithm's error rate decreases over successive iterations.

##  Local Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/dimssrmdn01/numerical-solver.git](https://github.com/dimssrmdn01/numerical-solver.git)
   cd numerical-solver

2. Install the required dependencies:
   Bash:
   pip install -r requirements.txt

4. Run the Streamlit application:
   Bash:
   streamlit run numerical_solver.py


## About The Project
This project was developed as an interactive learning tool and computational solver for Data Science and Statistics coursework at Institut Teknologi Sumatera. It aims to bridge the gap between theoretical numerical methods and practical software engineering.
