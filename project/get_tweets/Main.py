import sys
if sys.version_info[0] < 3:
  import got
else:
  import got3 as got
from datetime import date, timedelta
from dateutil.rrule import rrule, DAILY
import os
import pdb
import requests

def get_proxy():
    return requests.get("http://127.0.0.1:5010/get/").content

def delete_proxy(proxy):
    requests.get("http://127.0.0.1:5010/delete/?proxy={}".format(proxy))

def printTweet(descr, t):
  print(descr)
  print("Username: %s" % t.username)
  print("Retweets: %d" % t.retweets)
  print("Text: %s" % t.text)
  print("Mentions: %s" % t.mentions)
  print("Hashtags: %s\n" % t.hashtags)

if __name__ == '__main__':
  # # Example 1 - Get tweets by username
  # tweetCriteria = got.manager.TweetCriteria().setUsername('barackobama').setMaxTweets(1)
  # tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
  # printTweet("### Example 1 - Get tweets by username [barackobama]", tweet)

  # # Example 2 - Get tweets by query search
  # tweetCriteria = got.manager.TweetCriteria().setQuerySearch('europe refugees').setSince("2015-05-01").setUntil("2015-09-30").setMaxTweets(1)
  # tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
  # printTweet("### Example 2 - Get tweets by query search [europe refugees]", tweet)

  # # Example 3 - Get tweets by username and bound dates
  # tweetCriteria = got.manager.TweetCriteria().setUsername("barackobama").setSince("2015-09-10").setUntil("2015-09-12").setMaxTweets(1)
  # tweet = got.manager.TweetManager.getTweets(tweetCriteria)[0]
  # printTweet("### Example 3 - Get tweets by username and bound dates [barackobama, '2015-09-10', '2015-09-12']", tweet)

  save_dir = 'output'
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # Chosen according to https://fanpagelist.com/category/celebrities/view/list/sort/influence/
  # celebrity_usernames = ['barackobama', 'realdonaldtrump', 'britneyspears', 'thebeatles', 'nineinchnails',
  #                        'elvispresley', 'jtimberlake', 'rihanna', 'cher', 'michaeljackson',
  #                        'beyonce', 'kingjames', 'ladygaga', 'rogerfederer', 'liltunechi']
  celebrity_usernames = ['sfu', 'ubc']
  from_date = date(2018, 1, 1)
  to_date = date(2018, 10, 31)
  daily_fetch = 100

  # For testing
  # celebrity_usernames = ['barackobama']
  # from_date = date(2018, 11, 1)
  # to_date = date(2018, 11, 3)
  # daily_fetch = 100

  for celebrity_username in celebrity_usernames:
    keyword = "@%s"%(celebrity_username)

    sub_save_dir = os.path.join(save_dir, celebrity_username)
    if not os.path.exists(sub_save_dir):
      os.makedirs(sub_save_dir)

    for dt in rrule(freq=DAILY, dtstart=from_date, until=to_date):
      dt_str = dt.strftime("%Y-%m-%d")
      nxt_dt = dt + timedelta(days=1)
      nxt_dt_str = nxt_dt.strftime("%Y-%m-%d")
      tweetCriteria = got.manager.TweetCriteria() \
                      .setQuerySearch(keyword) \
                      .setSince(dt_str) \
                      .setUntil(nxt_dt_str) \
                      .setMaxTweets(daily_fetch)
      success = False
      retry_count = 5
      proxy = get_proxy()
      tweets = None
      while not success:
        if retry_count == 0:
          retry_count = 5
          delete_proxy(proxy)
          proxy = get_proxy()
        try:
          tweets = got.manager.TweetManager.getTweets(tweetCriteria, proxy=proxy)
        except Exception as e:
          print(e)
          retry_count -= 1
        else:
          success = True
          break

      f = open(os.path.join(sub_save_dir, dt_str), "w")
      for tweet in tweets:
        # pdb.set_trace()
        try:
          if keyword in tweet.mentions.lower().split():
            # print(tweet.username)
            # print(tweet.date)
            # print(tweet.text)
            # print('\n')
            f.write("%s\n"%tweet.text)
        except:
          # Tweet might inlude non-ASCII characters
          continue
      f.close()
