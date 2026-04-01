# Airline-Delay-Root-Cause-Project-Python-
Airline Delay Root-Cause Modeling

Built an end-to-end machine learning pipeline to model and explain flight delays across the U.S. using multi-year airline operations (BTS), NOAA weather data, and airport activity metrics (~800k records).

What I built:
	•	Integrated heterogeneous datasets (airline ops, weather, airport congestion)
	•	Engineered ~44 features capturing time-of-day, route, and environmental effects
	•	Trained and evaluated Logistic Regression, Random Forest, and Gradient Boosting
	•	Performed threshold tuning to analyze precision–recall tradeoffs under class imbalance (~80/20)

Key results:
	•	Random Forest provided the best balance (F1 ≈ 0.40, recall ≈ 0.58)
	•	Logistic Regression achieved highest recall (~0.62)
	•	Gradient Boosting had strongest ranking (ROC-AUC ≈ 0.70) but low recall at default thresholds

Key insight:
Flight delays are driven primarily by system-level factors, not isolated events.
Time-of-day (network congestion) and weather (precipitation, wind) consistently dominated across models.

Takeaway:
This project reinforced that effective ML is not just about accuracy—it’s about understanding tradeoffs and extracting actionable insight from complex systems.
