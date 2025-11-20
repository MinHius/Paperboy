from datetime import datetime


def is_trending():
    pass


def update_trend_score(stories: list) -> list:
    for story in stories:
        if story['created_at'] == datetime.now(): # Date only, remember to truncate timestamp
            story["trending"] = 10
        else:
            pass # Datetime subtraction here
    
    return stories
