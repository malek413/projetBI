# Business Intelligence Project: Hotel Booking Analysis

## 1. Project Description

This project focuses on the analysis of hotel booking data to extract actionable insights for the hospitality industry. It involves building a complete Business Intelligence solution, from data extraction and transformation (ETL) to data warehousing, dashboard creation, and predictive modeling. The primary dataset used is the "Hotel Booking Demand" dataset, which contains detailed information on reservations for a city hotel and a resort hotel between July 2015 and August 2017.

## 2. Project Objectives

The main goals of this project are:

* **Data Warehouse Design:** To design and implement an efficient data warehouse to store and organize hotel booking data using a dimensional model.
* **ETL Process Implementation:** To develop a robust ETL pipeline to extract data from the source, transform it into a suitable format, and load it into the data warehouse.
* **Dashboard Creation:** To design and develop interactive dashboards for visualizing key performance indicators (KPIs) and presenting analytical insights.
* **Machine Learning Application:** To apply machine learning techniques to the data warehouse to develop predictive models for addressing specific business problems (e.g., predicting booking cancellations).
* **Documentation and Presentation:** To thoroughly document the entire process and present the findings in a professional manner.

## 3. Dataset

* **Name:** Hotel Booking Demand
* **Source:** (Originally from an article by Nuno Antonio, Ana Almeida, and Luis Nunes, but often found on Kaggle and other data repositories. You can specify your exact source if known.)
* **Description:** Contains approximately 119,390 booking records for a city hotel and a resort hotel, with 32 variables covering booking details, customer information, stay specifics, and cancellation status.
* **Time Period:** Bookings made between July 2015 and August 2017.

## 4. Technologies Used

* **Data Visualization & Dashboards:** Microsoft Power BI
* **ETL (Extract, Transform, Load):** Talend Open Studio (or specified Talend version)
* **Data Warehouse:** MySQL
* **Machine Learning & Statistical Analysis:** R
* **Version Control:** Git & GitHub

## 5. Project Structure

A well-organized project structure is crucial for maintainability and collaboration. Below is a suggested structure.

## 6. Project Phases (Summary)

1.  **Analysis and Design (1 week):**
    * Dataset study and variable understanding.
    * Dimensional model design (star, snowflake, or constellation schema).
    * Definition of dimensions and facts.
2.  **ETL and Data Warehouse Implementation (2 weeks):**
    * Setup of the technical environment (MySQL).
    * Development of ETL scripts (Talend).
    * Implementation of the data loading process into the DWH.
    * Data integrity validation.
3.  **Analysis and Visualization (1 week):**
    * Development of interactive dashboards (Power BI).
    * Creation of visualizations for key KPIs.
4.  **Machine Learning (2 weeks):**
    * Formulation of a business problem (e.g., prediction of cancellations).
    * Data preparation for machine learning.
    * Algorithm selection and testing.
    * Model performance evaluation and selection of the best model.

## 7. Key Performance Indicators (KPIs) Analyzed

This project aims to analyze various KPIs, including but not limited to:

* **Booking Performance:**
    * Occupancy rate (by hotel type, month, season)
    * Average Revenue Per Available Room (RevPAR)
    * Average length of stay (by market segment)
    * Distribution of bookings by channel
* **Customer Behavior:**
    * Overall cancellation rate (and by segment)
    * Average lead time between booking and arrival
    * Customer retention rate
    * Geographic origin of customers (top countries)
* **Operational Efficiency:**
    * Discrepancy between requested and assigned room type
    * Average number of special requests per booking
    * Seasonal demand variations
* **Forecasting:**
    * Short-term occupancy rate prediction

## 8. How to Use/Run the Project

1.  **Prerequisites:**
    * MySQL Server
    * Talend Open Studio (or the version used)
    * R and RStudio (or your preferred R environment)
    * Power BI Desktop
2.  **Setup:**
    * Clone the repository: `git clone https://github.com/malek413/projetBI/Hotel-Booking-Analysis`
    * **Database:**
        * Create the MySQL database.
        * Run the schema creation scripts located in `data/`.
    * **ETL:**
        * Import Talend jobs from `etl/` into your Talend Open Studio.
        * Configure database connections within Talend jobs.
        * Run the main ETL job to populate the data warehouse from the raw data (ensure `data/raw/` contains the dataset).
    * **Machine Learning:**
        * Open R scripts from `Machine learning/` in RStudio.
        * Ensure all required R packages are installed (you might want to include a `requirements.R` or list packages here).
        * Set the working directory and database connection details within the scripts.
        * Run the scripts for analysis and model training.
    * **Dashboards:**
        * Open the `.pbix` files from `dashboards/` in Power BI Desktop.
        * Refresh data sources, ensuring Power BI can connect to your MySQL data warehouse or the provided data extracts.






* Malek Naffeti, Adem Makki
