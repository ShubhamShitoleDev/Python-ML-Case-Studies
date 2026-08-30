#################################################################################
#
#  Project Name  : Data Visualization & Analytics
#  Description   : Loads the Superstore sales dataset, performs
#                   exploratory data analysis and statistical summary
#                   reporting, analyzes sales/profit trends over time,
#                   breaks down performance by category/region/segment,
#                   examines the discount-vs-profit relationship, and
#                   assembles a single combined dashboard view.
#  Date          : 23-Aug-2026
#  Author        : Shubham Shitole
#
#################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Border = "-"*50


#################################################################################
#
# Function Name : load_data
# Input :         data_path
# Description :   Loads the Superstore CSV, parses Order Date / Ship Date
#                 as datetime, and prints an EDA summary
# Return Value :  Loaded DataFrame with parsed date columns
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def load_data(data_path):
    
    df = pd.read_csv(data_path, encoding="latin1")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    print("Shape of dataset : ", df.shape)
    print("First few records : ")
    print(df.head())

    print(Border)
    print("Column Info : ")
    df.info()

    print(Border)
    print("Missing Values Per Column : ")
    print(df.isnull().sum())

    print(Border)
    print("Duplicate Rows : ", df.duplicated().sum())

    return df


#################################################################################
#
# Function Name : kpi_summary
# Input :         Data
# Description :   Prints high-level business KPIs - total sales, total
#                 profit, overall profit margin, average discount, and
#                 total order count - the kind of top-line numbers a
#                 statistical report or dashboard would lead with
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def kpi_summary(Data):
    total_sales = Data["Sales"].sum()
    total_profit = Data["Profit"].sum()
    profit_margin = (total_profit / total_sales) * 100
    avg_discount = Data["Discount"].mean() * 100
    total_orders = Data["Order ID"].nunique()

    print(Border)
    print("KEY PERFORMANCE INDICATORS")
    print(Border)
    print(f"Total Sales        : ${total_sales:,.2f}")
    print(f"Total Profit        : ${total_profit:,.2f}")
    print(f"Overall Profit Margin : {profit_margin:.2f} %")
    print(f"Average Discount    : {avg_discount:.2f} %")
    print(f"Total Orders        : {total_orders:,}")


#################################################################################
#
# Function Name : time_trend_analysis
# Input :         Data
# Description :   Aggregates Sales and Profit by month and visualizes the
#                 trend over time
# Return Value :  monthly DataFrame (Sales, Profit indexed by month)
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def time_trend_analysis(Data):
    monthly = Data.set_index("Order Date").resample("M")[["Sales", "Profit"]].sum()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly.index, monthly["Sales"], label="Sales", marker="o")
    plt.plot(monthly.index, monthly["Profit"], label="Profit", marker="o")
    plt.title("Monthly Sales & Profit Trend")
    plt.xlabel("Month")
    plt.ylabel("Amount ($)")
    plt.legend()
    plt.grid()
    plt.show()

    return monthly


#################################################################################
#
# Function Name : category_region_analysis
# Input :         Data
# Description :   Breaks down Sales and Profit by Category, Sub-Category,
#                 and Region, visualizing each
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def category_region_analysis(Data):
    category_sales = Data.groupby("Category")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    print(Border)
    print("Sales & Profit by Category : ")
    print(category_sales)

    category_sales.plot(kind="bar", figsize=(8, 5))
    plt.title("Sales & Profit by Category")
    plt.ylabel("Amount ($)")
    plt.xticks(rotation=0)
    plt.show()

    subcat_sales = Data.groupby("Sub-Category")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    print(Border)
    print("Sales & Profit by Sub-Category : ")
    print(subcat_sales)

    subcat_sales["Sales"].plot(kind="bar", figsize=(12, 5), color="steelblue")
    plt.title("Sales by Sub-Category")
    plt.ylabel("Sales ($)")
    plt.show()

    region_sales = Data.groupby("Region")[["Sales", "Profit"]].sum().sort_values("Sales", ascending=False)
    print(Border)
    print("Sales & Profit by Region : ")
    print(region_sales)

    region_sales.plot(kind="bar", figsize=(8, 5))
    plt.title("Sales & Profit by Region")
    plt.ylabel("Amount ($)")
    plt.xticks(rotation=0)
    plt.show()


#################################################################################
#
# Function Name : top_products
# Input :         Data, top_n
# Description :   Finds and visualizes the top_n products by total sales
# Return Value :  DataFrame of the top_n products
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def top_products(Data, top_n=10):
    top = Data.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(top_n)

    print(Border)
    print(f"Top {top_n} Products by Sales : ")
    print(top)

    top.plot(kind="barh", figsize=(10, 6), color="seagreen")
    plt.title(f"Top {top_n} Products by Sales")
    plt.xlabel("Sales ($)")
    plt.gca().invert_yaxis()
    plt.show()

    return top


#################################################################################
#
# Function Name : discount_profit_analysis
# Input :         Data
# Description :   Visualizes the relationship between Discount and Profit,
#                 since heavy discounting is a common driver of lost profit
#                 margin in retail analytics
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def discount_profit_analysis(Data):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="Discount", y="Profit", data=Data, alpha=0.4)
    plt.title("Discount vs Profit")
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.show()

    print(Border)
    print("Average Profit by Discount Level : ")
    print(Data.groupby("Discount")["Profit"].mean())


#################################################################################
#
# Function Name : build_dashboard
# Input :         Data, monthly
# Description :   Assembles several of the individual analyses into one
#                 combined multi-panel dashboard figure, the way a
#                 Power BI / Tableau-style report would present several
#                 KPIs on a single screen
# Return Value :  None
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_dashboard(Data, monthly):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Marvellous Superstore Analytics Dashboard", fontsize=16)

    # Panel 1 : Monthly Sales Trend
    axes[0, 0].plot(monthly.index, monthly["Sales"], marker="o", color="steelblue")
    axes[0, 0].set_title("Monthly Sales Trend")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Panel 2 : Sales by Category
    category_sales = Data.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    axes[0, 1].bar(category_sales.index, category_sales.values, color="seagreen")
    axes[0, 1].set_title("Sales by Category")

    # Panel 3 : Sales by Region
    region_sales = Data.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    axes[1, 0].bar(region_sales.index, region_sales.values, color="darkorange")
    axes[1, 0].set_title("Sales by Region")

    # Panel 4 : Discount vs Profit
    axes[1, 1].scatter(Data["Discount"], Data["Profit"], alpha=0.3, color="indianred")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Discount vs Profit")
    axes[1, 1].set_xlabel("Discount")
    axes[1, 1].set_ylabel("Profit")

    plt.tight_layout()
    plt.show()


#################################################################################
#
# Function Name : main
# Input :         None
# Description :   Runs the full Data Visualization & Analytics pipeline
#                 end-to-end: load, KPI summary, time trend, category and
#                 region breakdowns, top products, discount/profit
#                 relationship, and a combined dashboard view
# Date :          23-Aug-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Step 1 : Load the Data Set")
    print(Border)

    df = load_data("Sample - Superstore.csv")

    print(Border)
    print("Step 2 : KPI Summary")
    print(Border)

    kpi_summary(df)

    print(Border)
    print("Step 3 : Time Trend Analysis")
    print(Border)

    monthly = time_trend_analysis(df)

    print(Border)
    print("Step 4 : Category & Region Analysis")
    print(Border)

    category_region_analysis(df)

    print(Border)
    print("Step 5 : Top Products")
    print(Border)

    top_products(df, top_n=10)

    print(Border)
    print("Step 6 : Discount vs Profit Analysis")
    print(Border)

    discount_profit_analysis(df)

    print(Border)
    print("Step 7 : Combined Dashboard")
    print(Border)

    build_dashboard(df, monthly)


if __name__ == "__main__":
    main()
