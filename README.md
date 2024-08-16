## SCAIO: Supply Chain AI Optimization Dashboard

SCAIO is an advanced AI-powered dashboard for visualizing, analyzing, and optimizing supply chain networks. It leverages cutting-edge machine learning techniques to provide actionable insights and optimize distribution strategies.

### Key Features

1. **Network Visualization**
   - Interactive map-based visualization of distribution centers and stores
   - Demand and inventory distribution heat maps
   - Store clustering based on location and demand patterns

2. **Traffic Analysis**
   - Traffic congestion visualization between distribution centers and stores
   - Detailed traffic statistics and congestion metrics

3. **Risk Assessment**
   - Risk graph highlighting vulnerable supply chain connections
   - Comprehensive risk statistics
   - Identification of high-risk distribution centers
   - Suggested optimal locations for DC relocation to mitigate risks

4. **Supply Chain Optimization**
   - Ant Colony Optimization (ACO) for efficient distribution path planning
   - Customizable optimization parameters
   - Visualization of optimized distribution routes

5. **Demand Forecasting**
   - Time series forecasting using SARIMAX models
   - Parallel processing for multi-store forecasts
   - Interactive charts for historical and forecasted demand
   - Performance rankings of distribution centers based on forecasted demand

6. **Cost Analysis**
   - Detailed breakdown of supply chain costs
   - Visual representation of cost components

7. **Relocation Recommendations**
   - AI-driven suggestions for optimal DC locations
   - Consideration of factors such as land cost, labor cost, and electricity cost
   - Risk factor assessment for potential locations

## Installation

1. Clone the repo `git clone https://github.com/ItsPranavz/sparkathon-scaio.git`
2. Install the below mentioned dependencies using `pip install`
3. Run `streamlit run dashboard.py`

## Dependencies

- streamlit
- numpy
- matplotlib
- scikit-learn
- networkx
- basemap
- statsmodels
- scipy
- pandas

---

SCAIO Dashboard: Empowering supply chain decisions with AI