# 🧪 Advanced Material Analysis System (with Gemini AI)

An interactive **Python-based material analysis assistant** that performs **chemical composition exploration, oxide correlation, classification, and AI-powered insights** using **Google Gemini AI**.
This tool is designed for **materials scientists, geologists, and researchers** who need quick, data-driven interpretation of oxide datasets from Excel files.

---

## 🚀 Features

### 🔍 Core Functionalities

* **Oxide Range Detection** – Displays min, max, and mean concentration ranges across samples.
* **Oxide Removal & Re-Normalization** – Exclude unwanted oxides and normalize remaining ones to 100%.
* **Cross-Sheet XY Plotting** – Compare two sheets from the same Excel file visually with regression and R² analysis.
* **Material Classification** – Identifies if a material is likely *Glass, Slag, Metal, or Ceramic* (AI + rule-based).
* **MnO Correlation Analysis** – Performs statistical and graphical correlation of MnO with other oxides.

### 🤖 AI-Driven Capabilities

* **AI Material Classification** – Uses **Google Gemini 1.5 Flash** to generate a detailed technical interpretation.
* **Predictive Insights** – Provides estimated *melting points, physical properties, chemical stability,* and *applications*.
* **LLM Query Interface** – Ask any question about the dataset in plain English.

### 🧭 Utility Commands

* List and switch between Excel sheets
* Display available regions/sites
* Generate visual insights and save plots automatically

---

## 🧩 Project Structure

```
📁 Advanced-Material-Analyzer/
│
├── data_analysis_assistant.py     # Main interactive script
├── data set for demo.xlsx         # Sample dataset (multi-sheet Excel)
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/Advanced-Material-Analyzer.git
cd Advanced-Material-Analyzer
```

### 2️⃣ Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate     # on Windows: venv\Scripts\activate
```

Then install required packages:

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Gemini API

Open `data_analysis_assistant.py` and replace:

```python
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

with your actual Gemini API key.

You can get it from [Google AI Studio](https://aistudio.google.com/).

---

## ▶️ How to Run

Run the main script:

```bash
python data_analysis_assistant.py
```

You’ll see the interactive console:

```
  ADVANCED MATERIAL ANALYSIS SYSTEM WITH GEMINI AI

CORE FEATURES:
  'ranges'       - Show oxide concentration ranges
  'remove'       - Remove oxides & view total change
  'renormalize'  - Renormalize data to 100%
  'xyplot'       - XY plot from two sheets
  'classify'     - Material classification
  'mno'          - MnO correlation analysis

CREATIVE FEATURES:
  'insights'     - AI predictive insights
  'regions'      - List available regions
  'sheets'       - List all sheets
  'switch <name>'- Switch to different sheet
  'ask <query>'  - Ask AI anything
  'quit'         - Exit
```

---

## 📊 Example Usage

**1️⃣ Show Oxide Ranges**

```bash
>>> ranges
```

**2️⃣ Remove Specific Oxides**

```bash
>>> remove
Enter oxide names to remove (comma-separated): FeO, MnO
```

**3️⃣ AI Classification**

```bash
>>> classify
```

**4️⃣ Generate XY Plot**

```bash
>>> xyplot
```

**5️⃣ Predictive Material Insights**

```bash
>>> insights
```

---

## 🧠 Technologies Used

* **Python 3.9+**
* **Pandas**, **NumPy**, **Matplotlib**, **Seaborn** – for data and visualization
* **SciPy** – for statistical analysis
* **Google Gemini 1.5 Flash** – for AI-based material reasoning and predictions

---

## 🧾 Sample Output

### Oxide Ranges

| Oxide | Min  | Max  | Mean | Range       |
| ----- | ---- | ---- | ---- | ----------- |
| SiO₂  | 45.2 | 72.1 | 59.8 | 45.2 – 72.1 |
| Al₂O₃ | 12.5 | 22.3 | 17.4 | 12.5 – 22.3 |
| FeO   | 2.1  | 8.7  | 5.4  | 2.1 – 8.7   |

---

## 📈 Example Generated Visuals

* **XY Scatter Plots with Trendlines**
* **Correlation Heatmaps**
* **MnO Scatter Grids**
* **Renormalized Composition Tables**

All plots are automatically **saved with timestamps**.

Would you like me to also generate a **`requirements.txt`** file (based on your script’s imports) so your GitHub repo looks complete and easy to run?
