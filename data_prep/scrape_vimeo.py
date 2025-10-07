import requests
import json


def get_vimeo_channel_videos(channel_id, access_token):
    """
    Scrapes all video URLs from a given Vimeo channel.

    Args:
        channel_id (str): The ID of the Vimeo channel.
        access_token (str): Your Vimeo API access token.

    Returns:
        list: A list of video URLs.
    """
    base_url = f"https://api.vimeo.com/channels/{channel_id}/videos"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.vimeo.*+json;version=3.4"
    }
    all_video_urls = []
    page = 1
    per_page = 100  # Max videos per page allowed by Vimeo API

    while True:
        params = {
            "page": page,
            "per_page": per_page
        }
        try:
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if not data.get("data"):
                break  # No more videos

            for video in data["data"]:
                # The 'link' field typically contains the video URL
                if "link" in video:
                    all_video_urls.append(video["link"])
                else:
                    print(f"Warning: 'link' not found for a video. Video data: {video}")

            # Check for the next page
            if "paging" in data and "next" in data["paging"] and data["paging"]["next"] is not None:
                page += 1
            else:
                break  # No more pages

        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            print(f"Response content: {response.content}")
            break
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error occurred: {e}")
            break
        except requests.exceptions.Timeout as e:
            print(f"Timeout error occurred: {e}")
            break
        except requests.exceptions.RequestException as e:
            print(f"An unexpected error occurred: {e}")
            break

    return all_video_urls


if __name__ == "__main__":
    # IMPORTANT: Replace with your actual Vimeo channel ID and access token
    # You can find your channel ID by navigating to your channel on Vimeo
    # and looking at the URL (e.g., https://vimeo.com/channels/YOUR_CHANNEL_ID)
    # Your access token should be a "personal access token" or "authenticated" token
    # generated in your Vimeo developer apps settings with appropriate scopes
    # (e.g., 'public' or 'private', 'video_files').
    VIMEO_CHANNEL_ID = "staffpicks"
    VIMEO_ACCESS_TOKEN = "5f88243e2ea474324c74df2e9d6dc297"

    if VIMEO_CHANNEL_ID == "YOUR_VIMEO_CHANNEL_ID" or VIMEO_ACCESS_TOKEN == "YOUR_VIMEO_API_KEY":
        print("Please replace 'YOUR_VIMEO_CHANNEL_ID' and 'YOUR_VIMEO_API_KEY' with your actual Vimeo channel ID and API key.")
    else:
        print(f"Scraping videos from Vimeo channel ID: {VIMEO_CHANNEL_ID}...")
        video_urls = get_vimeo_channel_videos(VIMEO_CHANNEL_ID, VIMEO_ACCESS_TOKEN)

        if video_urls:
            print(f"\nFound {len(video_urls)} video URLs:")
            for url in video_urls:
                print(url)
            # You can also save these URLs to a file
            # with open("vimeo_channel_video_urls.txt", "w") as f:
            #     for url in video_urls:
            #         f.write(url + "\n")
            # print("\nVideo URLs saved to vimeo_channel_video_urls.txt")
        else:
            print("No video URLs found or an error occurred.")
