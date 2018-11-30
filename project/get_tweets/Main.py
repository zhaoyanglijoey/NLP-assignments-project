import sys
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

def main():

	def printTweet(descr, t):
		print(descr)
		print("Username: %s" % t.username)
		print("Retweets: %d" % t.retweets)
		print("Text: %s" % t.text)
		print("Mentions: %s" % t.mentions)
		print("Hashtags: %s\n" % t.hashtags)

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

	tweetCriteria = got.manager.TweetCriteria().setQuerySearch("@realDonaldTrump").setSince("2018-10-01").setUntil("2018-10-07").setMaxTweets(5000)
	tweets = got.manager.TweetManager.getTweets(tweetCriteria)
	print("Tweets that mention Donald Trump from 2018-10-01 and 2018-10-07\n\n")
	for tweet in tweets:
	  print("@%s"%(tweet.username))
	  print(tweet.text.encode('utf-8'))
	  print('\n')

if __name__ == '__main__':
	main()
