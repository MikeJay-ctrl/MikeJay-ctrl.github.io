---
layout: post
title:  "A quick demo of Pandas with Music4ML "
date:   2017-12-28 19:00:24 +0000
category: "Data Science"
---



<br/><br/>
The Music4ML dataset contains a csv and an sqlite3.db file for your consumption. So in this post I thought i'd provide some insight on how to consume and the data with pandas.


The CSV file can be imported using pandas natve read_csv command as shown below 

	#This assumes your script is in the same location as the lyrics_data.csv file
    pd.read_csv('lyrics_data.csv')


You will see an output similar to below
<br/><br/>
![dataframe](/images/img/2018/1/pandas-csv-import-data.png)


Lets look at a few operations:

## View distribution by year

    df.groupby('year')['year'].count()

<br/><br/>
![year distribution](/images/img/2018/1/pandas-csv-year-dist.png)
<br/>
## Grab music from the 80s in ascending order

    df[(df['year'] > 1980) & (df['year'] <=1990) ].sort_values(by=['year'])

<br/><br/>
![RandB](/images/img/2018/1/pandas-csv-randb.png)
<br/>

## Grab all music with the tag RandB

    df[df['genre'] == 'RandB'].count()



## As I proceed with my latest NLP project I'll continue to post snippets and updates...
# Enjoy the new year!

