# MMOEA-DL_Paper_Implementation
Successfully Implemented the paper named as " A Deep Reinforcement Learning-Assisted  Multimodal Multiobjective Bilevel Optimization  Method for Multirobot Task Allocation  Yuanyuan Yu  , Qirong Tang  , Member, IEEE, Qingchao Jiang  and Qinqin Fan  , Senior Member, IEEE"

# Multi-Objective Evolutionary Algorithm with Deep RL for Multi-Robot Task Allocation (MRTA)

This project implements **MMOEA-DL**, a hybrid **Multi-Objective Evolutionary Algorithm (MOEA)** with **Deep Reinforcement Learning (DRL)** to solve **Multi-Robot Task Allocation (MRTA)** problems efficiently.  
It also provides **baseline algorithms**, a **multi-evaluation framework**, and **automated statistical analysis** for research-grade experiments.

---

## 🚀 Features
- **MMOEA-DL Implementation**: Hybrid EA + DRL for multi-objective optimization.
- **Baseline Algorithms**:
  - IMMODE-SA (EA + Simulated Annealing)
  - MMODE-CSCD (EA with Crowding Distance)
  - SS-MOPSO (Particle Swarm Optimization)
  - DSC-MOAGDE (Decision-Space Clustering MOEA)
- **Evaluation Pipeline**:
  - Hypervolume (HV), PSP, SR, AFE, AST metrics
  - Pareto front visualizations
  - Boxplots & statistical tests (Wilcoxon, Friedman)
  - Paper-ready LaTeX tables

---

## 📂 Project Structure
│-- main.py # MMOEA-DL implementation
│-- baselines.py # Baseline algorithm implementations
│-- multi_eval.py # Multi-algorithm evaluation & plotting
│-- evaluate.py # Single-algorithm evaluation (MMOEA-DL only)
│-- utils.py # Task generation & helper functions
│-- ea_upper.py # Evolutionary algorithm operators
│-- train_pointer.py # PointerNet training for DRL component
│-- requirements.txt # Python dependencies

---

## ⚙️ Installation
bash
# Clone the repo
git clone https://github.com/<your-username>/mmoea-dl-mrta.git
cd mmoea-dl-mrta

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🏃 Running Experiments
1. Run MMOEA-DL only
python evaluate.py
 -> Outputs results to results_summary.csv
 -> Generates metrics: HV, PSP, SR, AFE, AST
2. Compare with Baselines
python multi_eval.py
-> Runs MMOEA-DL + IMMODE-SA + MMODE-CSCD + SS-MOPSO + DSC-MOAGDE
-> Produces:
   multi_algo_results.csv → summary metrics
   hv_boxplots.png → hypervolume boxplots
   pareto_fronts.png → Pareto front comparison
   results_table.tex → paper-style summary table
   wilcoxon_table.tex → statistical significance table

📊 Outputs
| Metric  | Description                     |
| ------- | ------------------------------- |
| **HV**  | Hypervolume indicator           |
| **PSP** | Pareto Solution Percentage      |
| **SR**  | Success Rate                    |
| **AFE** | Average Function Evaluations    |
| **AST** | Average Solution Time (seconds) |

📈 Visualization Examples

Pareto Fronts

HV Boxplots

🧪 Statistical Analysis
Wilcoxon test: Pairwise algorithm significance test → wilcoxon_table.tex
