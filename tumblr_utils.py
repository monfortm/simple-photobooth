import os
import pytumblr

cons_k = os.environ["CONSUMER_KEY"]
cons_s = os.environ["CONSUMER_SECRET"]
oauth_tok = os.environ["OAUTH_TOKEN"]
oauth_sec = os.environ["OAUTH_SECRET"]
website_name = os.environ["WEBSITE_NAME"]

client = pytumblr.TumblrRestClient(
    cons_k,
    cons_s,
    oauth_tok,
    oauth_sec
)

bloginfo = client.blog_info(website_name)
blog_url = bloginfo.get("blog", {}).get("url")


def post_picture(pic_path):
    print()
    print("# Posting picture to {url}".format(url=blog_url))
    client.create_photo(
        website_name, state="published",
        tags=["testing", "ok"], data=[pic_path])
