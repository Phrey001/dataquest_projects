# dataquest_projects
## Deposit and accumulate projects from Dataquest for showcase

### Projects completed:
1) Python for Data Science: Fundamentals Part II: Profitable App Profiles for the App Store and Google Play Markets
2) Python for Data Science: Intermediate: Exploring Hacker News Posts
3) Pandas and NumPy Fundamentals: Exploring eBay Car Sales Data
4) Data Visualization Fundamentals: Finding Heavy Traffic Indicators on I-94
5) Storytelling Data Visualization and Information Design: Storytelling Data Visualization on Exchange Rates
6) Data Cleaning And Analysis: Clean And Analyse Employee Exit Surveys
7) Data Cleaning Project Walkthrough 1: Analysing NYC High School Data
8) Data Cleaning Project Walkthrough 2: Star Wars Survey
9) SQL Fundamentals: Analysing CIA FactBook Data Using SQL
10) Intermediate SQL For Data Analysis: Answering Business Questions Using SQL
11) Data Analysis In Business: Popular Data Science Questions
12) Statistics Fundamentals: Investigating Fandango Movie Ratings
13) 'Intermediate Statistics: Averages And Variability': Finding The Best Markets To Advertise Inc (Also Included PowerPoint Presentation In GitHub Folder)
14) 'Probabilities: Fundamentals': Mobile App For Lottery Addiction
15) Conditional Probabilities: Building A Spam Filter With Naive Bayes
16) 'Hypothesis Testing: Fundamentals': Winning Jeopardy
17) Machine Learning Fundamentals: Predicting Car Prices

### Key features demonstrated:

1) Python for Data Science: Fundamentals Part II: Profitable App Profiles for the App Store and Google Play Markets
- Went through a complete data science workflow:
  - Clarify goal of project
  - Collected relevant data
  - Cleaned and prepare data
    - Egs. Removed inaccurate data, duplicate data, non-english characters ..etc
  - Analysed data
- Example python skills:
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
- Example python skills:
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
- Example python skills:
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
- Example python skills:
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
    
5) Storytelling Data Visualization and Information Design: Storytelling Data Visualization on Exchange Rates
- Steps in project:
  - Started by exploring and cleaning the data
  - Brainstormed ideas for storytelling data visualizations and chose one
  - Planned (sketched) our data visualization
  - Coded the planned (sketched) data visualization
- Example python/design skills:
  - Use information design principles (familiarity and maximizing the data-ink ratio) to create better graphs for an audience.
  - Apply elements of a story and how to create storytelling data visualizations using Matplotlib.
  - Create visual patterns using Gestalt principles for design.
  - Guide the audience's attention with pre-attentive attributes.
  - Use Matplotlib built-in styles — eg. FiveThirtyEight style.
  - Demonstrated rolling mean (ie. moving average) computation in python pandas library.

6) Data Cleaning And Analysis: Clean And Analyse Employee Exit Surveys
- Example skills:
    - Explore the data and figure out how to prepare it for analysis
    - Correct some of the missing values
    - Drop any data not needed for our analysis
    - Rename our columns
    - Verify the quality of data
    - Create new columns (eg. feature engineering)
    - Combine the data
    - Handle the missing values in other column
    - Aggregate the data
    - Example tools for data cleaning: 
      - apply(), map(), applymap(), fillna(), dropna(), drop(), melt(), concat(), merge()

7) Data Cleaning Project Walkthrough 1: Analysing NYC High School Data
- Example skills:
    - Investigation into relationships between demographics and SAT scores.
    - Combined multiple datasets into a single, clean pandas dataframe using various joins technique (eg. df.merge(), left joins and inner joins)
    - Performed various data cleaning techniques to prepare data before proceed with analysis.
    - Identified correlations based on combinations of statistical computation and charting with scatterplot and line of best fit.
    - Performed desktop research to suggest potential explanations on identified correlations/trends.

8) Data Cleaning Project Walkthrough 2: Star Wars Survey
- Example skills:
    - Exploring and cleaning the data.
    - High level analysis, followed by analysis by segmentation for more granular insights.

9) SQL Fundamentals: Analysing CIA FactBook Data Using SQL
- Example SQL skills:
  - Connect Jupyter Notebook to our database file to run SQL codes with Python libraries sqlite3.
  - Summary Statistics
  - Subqueries

10) Intermediate SQL For Data Analysis: Answering Business Questions Using SQL
- Example SQL skills:
  - Subqueries
  - Multiple joins (to filter columns)
  - Set operations (example of set operations UNION, INTERSECT, EXCEPT to filter rows)
  - Aggregate functions

11) Data Analysis In Business: Popular Data Science Questions
- Example skills:
    - Explored the business context in which data science happens.

12) Statistics Fundamentals: Investigating Fandango Movie Ratings
- Steps in project / Example skills:
  - Started by exploring and define project objectives based on background context
  - Clean and transform data as needed, redefine project projectives if needed based on data suitability
  - Compare statistical properties of 2 separate year groups, by exploring and illustrate using:
    - Kernel Density Estimate (KDE) plots
    - Relative Frequency Tables (expressed in percentages)
    - Grouped Bar Charts (Comparing statistical summary)
  - Suggest conclusion to analysis.
 
13) 'Intermediate Statistics: Averages And Variability': Finding The Best Markets To Advertise In
- Example skills:
  - How to summarize distributions such as using the mean, and the median.
  - How to measure the variability of a distribution such as using quartile ranges.
  - Using various charts such as botplots and bar charts to visusalise distributions.
  - Segmenting dataset by different variables to explore potential market segments considering a balance of different metrics
  - Filtering data by identifying and removing outliers for further segmental analysis
  - Retrieving, identify and print specified sections of JSON file to present metadata purposes

14) 'Probabilities: Fundamentals': Mobile App For Lottery Addiction
- Example skills:
  - How to calculate theoretical probabilities (empirical probabilities don't apply in this project)
  - How to use probability rules to solve probability problems
  - How to use combinations (permutations don't apply to this project)
  
15) Conditional Probabilities: Building A Spam Filter With Naive Bayes
- Example skills:
    - Apply the Naive Bayes algorithm with additive smoothing to classify messages to output binominal results (spam or non-spam).
        - Assign probabilities to events based on certain conditions by using conditional probability rules.
        - Assign probabilities to events based on whether they are in relationship of statistical independence or not with other events.
        - Assign probabilities to events based on prior knowledge by using Bayes' theorem.
        - Create a spam filter for SMS messages using the multinomial Naive Bayes algorithm.
        
16) 'Hypothesis Testing: Fundamentals': Winning Jeopardy
- Example skills applied:
    - Applied a chi-square test to identify if any significant statistical differences in categorial data scoped in project.
        - Processing and preparing the data to get the components of the test, before applying the formula.
        
17) Machine Learning Fundamentals: Predicting Car Prices
- Example skills applied:
    - Applied KNN machine-learning algorithm (k-nearest neighbors algorithm) and cross-validate with training/test sets using holdout validation approach (specific condition of k-fold cross validation where number of partition is 2) on both univariate and multivariate models.
    - Illustrated how k-fold cross validation with different number of partions could potentially be carried out, by applying it on a univariate model.