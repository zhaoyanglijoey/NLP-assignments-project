#!/bin/bash
# Group daily tweets into monthly tweets

celebrity_usernames="barackobama realdonaldtrump britneyspears thebeatles nineinchnails"
months="2018-01 2018-02 2018-03 2018-04 2018-05 2018-06 2018-07 2018-08 2018-09 2018-10"

for celebrity in $celebrity_usernames
do
  for month in $months
  do
    cat output/$celebrity/$month-* > output/$celebrity/$month
    rm output/$celebrity/$month-*
  done
done