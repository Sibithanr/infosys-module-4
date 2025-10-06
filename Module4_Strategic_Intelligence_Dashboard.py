# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns


# # 1. Page Config
# st.set_page_config(page_title="Strategic Intelligence Dashboard", layout="wide")
# st.title("Strategic Intelligence Dashboard")


# # 2. Generate Data

# np.random.seed(42)
# dates = pd.date_range(start="2023-01-01", periods=100)
# competitors = ["AlphaCorp", "BetaTech", "GammaSoft"]
# sectors = ["AI", "Cloud", "Fintech"]
# keywords = ["AI", "Cloud", "Fintech"]

# # Competitor data
# data = {
#     "date": np.tile(dates, len(competitors)),
#     "competitor": np.repeat(competitors, len(dates)),
#     "sector": np.random.choice(sectors, len(dates) * len(competitors)),
#     "sentiment_score": np.random.uniform(-1, 1, len(dates) * len(competitors)),
#     "mentions": np.random.randint(10, 100, len(dates) * len(competitors)),
# }
# df = pd.DataFrame(data)
# kw_rows = []
# for comp in competitors:
#     for kw in keywords:
#         counts = np.random.randint(5, 50, len(dates))
#         for i, date in enumerate(dates):
#             kw_rows.append({"date": date, "competitor": comp, "keyword": kw, "count": counts[i]})
# kw_df = pd.DataFrame(kw_rows)


# # 3. Sidebar Filters

# st.sidebar.header("Filters")
# selected_competitors = st.sidebar.multiselect("Select Competitors", competitors, default=competitors)
# selected_sectors = st.sidebar.multiselect("Select Sectors", sectors, default=sectors)
# date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

# # Apply filters
# mask = (
#     df["competitor"].isin(selected_competitors)
#     & df["sector"].isin(selected_sectors)
#     & (df["date"] >= pd.to_datetime(date_range[0]))
#     & (df["date"] <= pd.to_datetime(date_range[1]))
# )
# filtered_df = df[mask]

# # Keyword filter
# kw_mask = (
#     kw_df["competitor"].isin(selected_competitors)
#     & (kw_df["date"] >= pd.to_datetime(date_range[0]))
#     & (kw_df["date"] <= pd.to_datetime(date_range[1]))
# )
# filtered_kw_df = kw_df[kw_mask]


# # 4. KPIs

# st.subheader("Key Metrics")
# col1, col2, col3, col4 = st.columns(4)
# col1.metric("Total Competitors", len(selected_competitors))
# col2.metric("Avg Sentiment", f"{filtered_df['sentiment_score'].mean():.2f}")
# col3.metric("Total Mentions", filtered_df['mentions'].sum())
# alerts = filtered_df[filtered_df["sentiment_score"] < -0.7]
# col4.metric("Alerts", len(alerts))

# # 5. Charts
# st.subheader(" Competitor Sentiment Trajectories")
# fig, ax = plt.subplots(figsize=(10,5))
# sns.lineplot(data=filtered_df, x="date", y="sentiment_score", hue="competitor", ax=ax)
# st.pyplot(fig)

# st.subheader("Mentions Over Time")
# fig2, ax2 = plt.subplots(figsize=(10,5))
# sns.lineplot(data=filtered_df, x="date", y="mentions", hue="competitor", ax=ax2)
# st.pyplot(fig2)

# st.subheader("Keyword Trend Evolution")
# fig3, ax3 = plt.subplots(figsize=(10,5))
# if not filtered_kw_df.empty:
#     kw_pivot = filtered_kw_df.pivot_table(index="date", columns="keyword", values="count", aggfunc="sum")
#     sns.lineplot(data=kw_pivot, ax=ax3)
#     st.pyplot(fig3)

# # 6. Alerts Table  Download
# st.subheader(" Alerts")
# if not alerts.empty:
#     st.dataframe(alerts)
#     csv_alerts = alerts.to_csv(index=False).encode('utf-8')
#     st.download_button("Download Alerts CSV", data=csv_alerts, file_name="alerts.csv")
# else:
#     st.success("No critical alerts")

# # 7. Filtered Data Table  Download
# st.subheader("Filtered Dataset")
# st.dataframe(filtered_df)
# csv_data = filtered_df.to_csv(index=False).encode('utf-8')
# st.download_button("Download Filtered Data CSV", data=csv_data, file_name="filtered_data.csv")



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 1. Page Config
st.set_page_config(page_title="Strategic Intelligence Dashboard", layout="wide")
st.title("Strategic Intelligence Dashboard with Forecasts & Insights")

# 2. Generate Data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100)
competitors = ["AlphaCorp", "BetaTech", "GammaSoft"]
sectors = ["AI", "Cloud", "Fintech"]
keywords = ["AI", "Cloud", "Fintech"]

# Competitor dataset
data = {
    "date": np.tile(dates, len(competitors)),
    "competitor": np.repeat(competitors, len(dates)),
    "sector": np.random.choice(sectors, len(dates) * len(competitors)),
    "sentiment_score": np.random.uniform(-1, 1, len(dates) * len(competitors)),
    "mentions": np.random.randint(10, 100, len(dates) * len(competitors)),
}
df = pd.DataFrame(data)

# Keyword dataset
kw_rows = []
for comp in competitors:
    for kw in keywords:
        counts = np.random.randint(5, 50, len(dates))
        for i, date in enumerate(dates):
            kw_rows.append({"date": date, "competitor": comp, "keyword": kw, "count": counts[i]})
kw_df = pd.DataFrame(kw_rows)

# 3. Sidebar Filters
st.sidebar.header("Filters")
selected_competitors = st.sidebar.multiselect("Select Competitors", competitors, default=competitors)
selected_sectors = st.sidebar.multiselect("Select Sectors", sectors, default=sectors)
date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

# Apply filters
mask = (
    df["competitor"].isin(selected_competitors)
    & df["sector"].isin(selected_sectors)
    & (df["date"] >= pd.to_datetime(date_range[0]))
    & (df["date"] <= pd.to_datetime(date_range[1]))
)
filtered_df = df[mask]

# Keyword filter
kw_mask = (
    kw_df["competitor"].isin(selected_competitors)
    & (kw_df["date"] >= pd.to_datetime(date_range[0]))
    & (kw_df["date"] <= pd.to_datetime(date_range[1]))
)
filtered_kw_df = kw_df[kw_mask]

# 4. KPIs

st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Competitors", len(selected_competitors))
col2.metric("Avg Sentiment", f"{filtered_df['sentiment_score'].mean():.2f}")
col3.metric("Total Mentions", filtered_df['mentions'].sum())
alerts = filtered_df[filtered_df["sentiment_score"] < -0.7]
col4.metric("Alerts", len(alerts))

# 5. Charts
st.subheader("Competitor Sentiment Trajectories")
fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x="date", y="sentiment_score", hue="competitor", ax=ax)
st.pyplot(fig)

st.subheader("Mentions Over Time")
fig2, ax2 = plt.subplots(figsize=(10,5))
sns.lineplot(data=filtered_df, x="date", y="mentions", hue="competitor", ax=ax2)
st.pyplot(fig2)

st.subheader("Keyword Trend Evolution")
fig3, ax3 = plt.subplots(figsize=(10,5))
if not filtered_kw_df.empty:
    kw_pivot = filtered_kw_df.pivot_table(index="date", columns="keyword", values="count", aggfunc="sum")
    sns.lineplot(data=kw_pivot, ax=ax3)
    st.pyplot(fig3)


# 6. Forecasting Section

st.subheader(" Forecasting Trends")
forecast_comp = st.selectbox("Select a competitor for forecasting", selected_competitors)

comp_data = filtered_df[filtered_df["competitor"] == forecast_comp].set_index("date")

if not comp_data.empty:
    # Mentions Forecast
    st.markdown(f"**Mentions Forecast for {forecast_comp}**")
    ts_mentions = comp_data["mentions"].asfreq("D").interpolate()
    model_mentions = ExponentialSmoothing(ts_mentions, trend="add", seasonal=None)
    fit_mentions = model_mentions.fit()
    forecast_mentions = fit_mentions.forecast(30)

    fig4, ax4 = plt.subplots(figsize=(10,5))
    ts_mentions.plot(ax=ax4, label="Historical")
    forecast_mentions.plot(ax=ax4, label="Forecast", color="red")
    ax4.set_title(f"{forecast_comp} - Mentions Forecast (Next 30 days)")
    ax4.legend()
    st.pyplot(fig4)

    # Sentiment Forecast
    st.markdown(f"**Sentiment Forecast for {forecast_comp}**")
    ts_sent = comp_data["sentiment_score"].asfreq("D").interpolate()
    model_sent = ExponentialSmoothing(ts_sent, trend="add", seasonal=None)
    fit_sent = model_sent.fit()
    forecast_sent = fit_sent.forecast(30)

    fig5, ax5 = plt.subplots(figsize=(10,5))
    ts_sent.plot(ax=ax5, label="Historical")
    forecast_sent.plot(ax=ax5, label="Forecast", color="orange")
    ax5.set_title(f"{forecast_comp} - Sentiment Forecast (Next 30 days)")
    ax5.legend()
    st.pyplot(fig5)

else:
    st.warning("No data available for forecasting.")

# 7. Additional Charts
st.subheader(" Additional Insights")

# Sentiment Distribution
st.markdown("### Sentiment Distribution by Competitor")
fig6, ax6 = plt.subplots(figsize=(8,5))
sns.boxplot(data=filtered_df, x="competitor", y="sentiment_score", ax=ax6)
st.pyplot(fig6)

# Mentions Share
st.markdown("### Mentions Share Across Competitors")
mentions_share = filtered_df.groupby("competitor")["mentions"].sum()
fig7, ax7 = plt.subplots(figsize=(6,6))
ax7.pie(mentions_share, labels=mentions_share.index, autopct='%1.1f%%', startangle=90)
ax7.axis("equal")
st.pyplot(fig7)

# Heatmap (Competitor vs Sector)
st.markdown("### Avg Sentiment Heatmap (Competitor vs Sector)")
pivot_heatmap = filtered_df.pivot_table(index="competitor", columns="sector", values="sentiment_score", aggfunc="mean")
fig8, ax8 = plt.subplots(figsize=(7,5))
sns.heatmap(pivot_heatmap, annot=True, cmap="coolwarm", center=0, ax=ax8)
st.pyplot(fig8)

# Rolling Average Sentiment
st.markdown("### Rolling Avg Sentiment (7-day)")
fig9, ax9 = plt.subplots(figsize=(10,5))
for comp in selected_competitors:
    temp = filtered_df[filtered_df["competitor"] == comp].set_index("date").sort_index()
    temp["sentiment_roll"] = temp["sentiment_score"].rolling(7).mean()
    ax9.plot(temp.index, temp["sentiment_roll"], label=comp)
ax9.legend()
st.pyplot(fig9)

# Cumulative Mentions
st.markdown("### Cumulative Mentions")
fig10, ax10 = plt.subplots(figsize=(10,5))
for comp in selected_competitors:
    temp = filtered_df[filtered_df["competitor"] == comp].set_index("date").sort_index()
    ax10.plot(temp.index, temp["mentions"].cumsum(), label=comp)
ax10.legend()
st.pyplot(fig10)

# Keyword Share
st.markdown("### Keyword Share per Competitor")
kw_share = filtered_kw_df.groupby(["competitor","keyword"])["count"].sum().unstack(fill_value=0)
fig11, ax11 = plt.subplots(figsize=(8,5))
kw_share.plot(kind="bar", stacked=True, ax=ax11)
ax11.set_ylabel("Keyword Counts")
st.pyplot(fig11)

# Keyword Co-occurrence Heatmap
st.markdown("### Keyword Co-occurrence Heatmap")
if not filtered_kw_df.empty:
    co_occurrence = filtered_kw_df.groupby(["date","competitor","keyword"])["count"].sum().unstack(fill_value=0)
    co_matrix = co_occurrence.corr()
    fig12, ax12 = plt.subplots(figsize=(6,5))
    sns.heatmap(co_matrix, annot=True, cmap="Blues", ax=ax12)
    st.pyplot(fig12)

# 8. Alerts Table  Download
st.subheader("Alerts")
if not alerts.empty:
    st.dataframe(alerts)
    csv_alerts = alerts.to_csv(index=False).encode('utf-8')
    st.download_button("Download Alerts CSV", data=csv_alerts, file_name="alerts.csv")
else:
    st.success("No critical alerts")

# 9. Filtered Data Table Download
st.subheader("Filtered Dataset")
st.dataframe(filtered_df)
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Data CSV", data=csv_data, file_name="filtered_data.csv")
