# Real-Time Industry Insight & Strategic Intelligence System

## Overview
A comprehensive **interactive dashboard** for tracking, analyzing, and forecasting industry intelligence across multiple technology sectors using competitor, news, and social media data. This system provides actionable insights for strategic decision-making, combining **real-time analytics, forecasting, alerts, and data visualization**.

---

## 🚀 Features

### Core Functionality
- **Domain & Competitor Selection:** Filter by sectors and competitors.
- **Interactive Dashboard:** KPIs, sentiment trends, mentions, alerts, and keyword analytics.
- **Forecasting:** Predict sentiment and mentions using **Exponential Smoothing**.
- **Alerts:** Automated detection of negative sentiment and mention spikes.
- **Data Export:** Download filtered datasets and alerts as CSV.
- **Themes:** Toggle between **dark/light modes** for optimal viewing.

### Analytical Insights
- **Competitor Sentiment Trajectories:** Track sentiment trends over time.
- **Mentions Analysis:** Historical and cumulative mentions visualization.
- **Keyword Trends:** Track keyword mentions and co-occurrences.
- **Heatmaps & Boxplots:** Compare sentiment across competitors and sectors.
- **Rolling Averages:** Identify trends with 7-day moving averages.

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit** for interactive dashboard
- **Matplotlib & Seaborn** for charts
- **Interactive Filters & Downloads** using Streamlit widgets

### Backend / Data Processing
- **Python** with Pandas & NumPy
- **Statsmodels** for time-series forecasting
- **Exponential Smoothing** for mentions & sentiment prediction
- **CSV Parsing** for real-time data ingestion

### Version Control
- Git & GitHub for source code management


---

## 🏗️ Installation & Setup

### Prerequisites
- Python 3.8+  
- Pip  

### Quick Start
1. Clone the repository:
bash
git clone https://github.com/Sibithanr/infosys-module-4
cd infosys-module-4
Create & activate a virtual environment:

bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
python -m streamlit run Module4_Strategic_Intelligence_Dashboard.py
Open http://localhost:8501 in your browser.

📊 Data Format
Competitor CSV
bash
Copy code
date,competitor,sector,sentiment_score,mentions
2025-01-01,AlphaCorp,AI,0.75,50
Keyword CSV
bash
Copy code
date,competitor,keyword,count
2025-01-01,AlphaCorp,AI,25
Sentiment scores are automatically interpolated if missing.

🔄 Features in Action
KPIs: Total competitors, average sentiment, total mentions, alerts.

Charts: Line plots, heatmaps, pie charts, rolling averages, cumulative mentions.

Forecasting: 30-day prediction of mentions & sentiment per competitor.

Alerts Table: Automatic detection of negative sentiment or spikes in mentions.

Downloadable Data: Export filtered datasets and alerts as CSV.

🧪 Testing
Verify sidebar filters and date ranges work correctly.

Confirm charts and forecasts render correctly for each competitor.

Test alerts generation and CSV download functionality.

Check rolling averages, cumulative mentions, and heatmaps display as expected.

🎨 Customization
Add new competitors or sectors by updating the CSV files.

Extend forecasting for new metrics via forecasting.py.

Modify chart styles in Module4_Strategic_Intelligence_Dashboard.py using Matplotlib or Seaborn.

📝 License
MIT License. See LICENSE file for details.

🤝 Author
Sibitha Namakkal Ravikumar
