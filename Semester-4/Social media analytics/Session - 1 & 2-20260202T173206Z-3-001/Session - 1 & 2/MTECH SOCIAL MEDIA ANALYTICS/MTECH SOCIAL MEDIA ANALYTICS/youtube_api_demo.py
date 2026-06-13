from googleapiclient.discovery import build

# Step 3: Add your API key
API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxx"

youtube = build("youtube", "v3", developerKey=API_KEY)

# Step 4: Search for a product review video
request = youtube.search().list(
    q="Samsung Galaxy S23 review",
    part="snippet",
    type="video",
    maxResults=1
)

response = request.execute()
video_id = response["items"][0]["id"]["videoId"]

print("Video ID:", video_id)

# Step 5: Fetch user comments
comments_request = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    maxResults=10
)

comments_response = comments_request.execute()

print("\nUser Comments:\n")

for item in comments_response["items"]:
    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
    print("-", comment)

# Step 6: Engagement metrics
video_stats = youtube.videos().list(
    part="statistics",
    id=video_id
).execute()

stats = video_stats["items"][0]["statistics"]

print("\nEngagement Stats:")
print("Views:", stats["viewCount"])
print("Likes:", stats.get("likeCount", 0))
print("Comments:", stats["commentCount"])
