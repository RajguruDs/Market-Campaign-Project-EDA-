# Marketing Campaign Analysis – End-to-End Project

**Author:** Rajguru  
**GitHub Profile:** https://github.com/RajguruDs  
**Published Dashboard:** [Interactive Tableau Dashboard](https://public.tableau.com/views/MarketingCampaignAnalysis_17630457873700/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Project Overview  
This project performs an in-depth analysis of a marketing campaign dataset to uncover insights into customer behaviour, segment the customer base, and support data-driven marketing strategies.

---

## Dataset Overview  
The dataset contains customer attributes including:  
- Demographics (Age, Education, Marital Status)  
- Financial data (Income)  
- Purchase history across different product categories  
- Marketing campaign response information  

---

## Project Goals  
- Understand customer profiles and purchase behaviour  
- Identify trends in marketing campaign responses  
- Segment customers for targeted marketing strategies  
- Build a clear, visually engaging dashboard for decision-making  

---

## Methodology & Workflow  
### 1. Data Inspection  
- Loaded raw data using Pandas  
- Inspected for missing values, data types, distributions  
- Initial exploratory checks  

### 2. Data Cleaning  
- Handled missing or null values (imputation / removal)  
- Converted monetary and categorical fields to consistent formats  
- Filtered out outliers and invalid records  
- Ensured data was analysis-ready  

### 3. Exploratory Data Analysis (EDA)  
- Visualised distributions (age, income, product spend)  
- Analysed relationships between features and campaign response  
- Created summary tables and visualisations (histograms, scatter plots, heatmaps)  
- Drew initial insights on customer behaviour  

### 4. Customer Segmentation & Feature Engineering  
- Developed meaningful segments (e.g., Recency, Frequency, Monetary value)  
- Encoded categorical variables, standardised numerical features  
- Identified key features impacting campaign response  

### 5. Data Visualization & Dashboard  
- Built an interactive dashboard using Tableau  
- Included key performance indicators (KPIs) such as conversion rate, total campaigns responded, channel effectiveness  
- Enabled filtering by customer demographics, campaign type, region  
- Provided clear visuals for stakeholders to interpret insights  
  > **See the live dashboard link above.**  

---

## Key Insights  
- High income does *not* always correlate with higher campaign responsiveness  
- Younger customers showed higher spending on certain product categories  
- Married and divorced customers had higher response rates in selected campaigns  
- Certain marketing channels significantly out-performed others in conversion  
- Recommendations: Focus on the high-response segments & optimise budget allocation across channels  

---

## Files Included  
- `Market Campaign Project.ipynb` – detailed notebook with cleaning, EDA, feature engineering  
- `marketing_data.csv` – original raw dataset  
- `requirements.txt` – list of Python libraries used  
- **Live dashboard link** – see above  

---

## How to Use  
1. Clone or download this repository.  
2. Open `Market Campaign Project.ipynb` in Jupyter Notebook or an equivalent environment.  
3. Run the notebook cells step by step to follow the cleaning and analysis logic.  
4. Explore the dashboard link for interactive visual insights.  

---

## Tools & Technologies Used  
- **Programming / Data Tools:** Python, Pandas, NumPy, Scikit-learn   
- **Visualization / BI Tool:** Tableau  
- **Notebook Environment:** Jupyter Notebook  

---

## Future Scope  
- Add machine learning model(s) (e.g., classification of likely responders)  
- Integrate SQL-based extraction for large-scale data workflows  
- Deploy dashboard in a web application or with automated refresh  
- Include deeper customer lifetime value (CLV) modelling or time-series analysis  

---

## Author  
Rajguru — B.Sc. IT student | Aspiring Data Scientist  
Feel free to connect via [LinkedIn](https://www.linkedin.com/in/rajguru-mathiyalagan-63b921244/) or email at rajguru21.ds@gmail.com
