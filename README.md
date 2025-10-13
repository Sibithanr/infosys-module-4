# Real-Time Industry Insight & Strategic Intelligence System

## Overview
The **Real-Time Industry Insight & Strategic Intelligence System** is an interactive dashboard built using **Python** and **Streamlit** that provides actionable insights into competitor performance, sentiment trends, and keyword dynamics across multiple sectors. The system also includes forecasting features to predict mentions and sentiment trends, helping businesses make data-driven strategic decisions.

---

## Features
- **Key Metrics (KPIs):** Total competitors, average sentiment, total mentions, and alerts for negative sentiment.  
- **Competitor Analytics:** Line charts tracking sentiment and mentions over time.  
- **Keyword Trends:** Trend analysis for selected keywords across competitors.  
- **Forecasting:** Exponential Smoothing forecasts for competitor mentions and sentiment for the next 30 days.  
- **Visual Insights:**  
  - Sentiment distribution by competitor  
  - Mentions share across competitors  
  - Heatmaps for competitor vs sector sentiment  
  - Rolling averages and cumulative mentions  
  - Keyword share and co-occurrence heatmaps  
- **Data Download:** Download filtered datasets and alerts in CSV format.  

---

## Tech Stack
- **Programming Language:** Python  
- **Libraries & Frameworks:** Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Statsmodels  
- **Version Control:** Git & GitHub  

---

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RealTime-Industry-Insight.git
   cd RealTime-Industry-Insight
Create a virtual environment (optional but recommended):

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
Install required packages:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Usage
Use the sidebar filters to select competitors, sectors, and date ranges.

Explore KPIs and charts for sentiment and mentions insights.

Select a competitor for forecasting trends.

Download alerts and filtered datasets for further analysis.

Screenshots
(Optional: Add screenshots of your dashboard here)

License
This project is licensed under the MIT License. See the LICENSE file for details.

Author
Sibitha Namakkal Ravikumar
