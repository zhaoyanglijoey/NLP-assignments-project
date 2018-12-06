#!/bin/bash
# Group daily tweets into monthly tweets

celebrity_usernames="sfu ubc"
months="2018-01 2018-02 2018-03 2018-04 2018-05 2018-06 2018-07 2018-08 2018-09 2018-10"

echo $1

if [ $1 = "--monthly" ] || [ $1 = "-m" ]
then
  echo "Grouping tweets by months..."
  for celebrity in $celebrity_usernames
  do
    for month in $months
    do
      cat output/$celebrity/$month-* > output/$celebrity/$month
      rm output/$celebrity/$month-*
    done
  done
elif [ $1 = "--weekly" ] || [ $1 = "-w" ]
then
  echo "Grouping tweets by weeks..."
  for celebrity in $celebrity_usernames
  do
    count=0
    group_size=7
    weekly_tweet=""
    first_day=""
    last_day=""
    for daily_tweet in `ls output/$celebrity | sort`
    do
      count=$((count+1))
      if [ -z $first_day ]
      then
        first_day="$daily_tweet"
      fi
      if [ "$count" -eq "$group_size" ]
      then
        weekly_tweet="$weekly_tweet output/$celebrity/$daily_tweet"
        last_day="$daily_tweet"
        cat $weekly_tweet > output/$celebrity/${first_day}_${last_day}
        rm $weekly_tweet
        count=0
        weekly_tweet=""
        first_day=""
        last_day=""
      else
        weekly_tweet="$weekly_tweet output/$celebrity/$daily_tweet"
      fi
    done
  done
fi
