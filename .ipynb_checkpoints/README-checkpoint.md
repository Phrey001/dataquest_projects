# dataquest_projects
## Deposit and accumulate projects from Dataquest for showcase

### Projects completed:
1) Python for Data Science: Fundamentals Part II: Profitable App Profiles for the App Store and Google Play Markets
2) Python for Data Science: Intermediate: Exploring Hacker News Posts
3) Pandas and NumPy Fundamentals: Exploring eBay Car Sales Data
4) Data Visualization Fundamentals: Finding Heavy Traffic Indicators on I-94

### Key features demonstrated:
1) Python for Data Science: Fundamentals Part II: Profitable App Profiles for the App Store and Google Play Markets
- Went through a complete data science workflow:
  - Clarify goal of project
  - Collected relevant data
  - Cleaned and prepare data
    - Egs. Removed inaccurate data, duplicate data, non-english characters ..etc
  - Analysed data
- Example python skills used:
  - Opened and load csv file
  - Used objects and functions
    - Egs. lists, dictionaries, loops, nested loops, conditional statements ..etc
  - Segmented groups for analysis
  - Built a Frequency table
  - Coded in Jupyter Notebook
2) Python for Data Science: Intermediate: Exploring Hacker News Posts
- Steps in project:
  - Set a goal for the project
  - Collected and sorted the data
  - Reformatted and cleaned data
  - Analysed data
- Example python skills used:
  - Working with strings
    - Egs. transform to lower cases with .lower(), filter with .startswith()
  - Object-oriented programming
    - Egs. Using classes and methods like print using template.format(); working with list of lists, dictionaries and sorting/manipulating elements, keys, values within ..etc
  - Dates and times
    - dt.datetime.strptime functions to parse datetime objects
    - dt.datetime.strftime functions to format datetime objects
3) Pandas and NumPy Fundamentals: Exploring eBay Car Sales Data
- Steps in project:
  - Practiced applying a variety of pandas methods to explore and understand a data set on car listings
- Example python skills used:
  - Boolean indexing with NumPy
    - Egs. Used NumPy's arrays of Boolean to filter select dataframe rows
  - Exploring data with Pandas/NumPy, Egs.
    - Import and load csv file with Pandas to create a dataframe for further work
    - Exploratory functions
      - Egs. df.info(), .describe(), df.head(), df.tail(), df.columns(), df.value_counts(), Series.sort_index()/.sort_values() /.unique()/.shape() ..etc
    - Clean column names by renaming/replacing to lowercase, snakecase, strip white space if necessary ..etc
    - Datatype transformations (egs. astype(int) to change datatype to integer)
    - Dataframe manipulations (egs. df.drop() to drop columns, remove outliers / filter select data rows using boolean indexing ..etc)
    - Objects/methods chaining (egs. Series..value_counts().sort_index() ..etc)
4) Data Visualization Fundamentals: Finding Heavy Traffic Indicators on I-94
- Steps in project:
  - Exploratory data visualization: build graphs for ourselves to explore data and find patterns.
- Example python skills used:
  - Data visualisation libraries such as matplotlib, seaborn
    - Visualise time series data with line plots
    - Visualise correlations with scatter plots
    - Visualise frequency distributions with bar plots and histograms
    - Speed up our exploratory data visualization workflow with the pandas library
      - ie. direct plotting with pandas library code instead of using matplotlib
    - Compare graphs using grid charts (ie. multiple subplots on one grid chart) for ease of presentation
    - Used seaborn to slightly enhance plot aesthetics only
      - Seaborn has capability to plot on a single chart featuring multiple variables with qualities such as shape, color shades (ie. hue), sizes ..etc, but did not use it in this project
    - Datetime objects manipulation using pandas, egs. pd.to_datetime()