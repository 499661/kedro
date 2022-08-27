import requests

def ttest_result(webhook_url:str, content:str, title:str, challenger_url: str, action:str , color:str="000000") -> int:
    """
      - Send a teams notification to the desired webhook_url
      - Returns the status code of the HTTP request
        - webhook_url : the url you got from the teams webhook configuration
        - content : your formatted notification content
        - title : the message that'll be displayed as title, and on phone notifications
        - challenger_url: url of the challenger model trained
        - action : message for the action to be taken
        - color (optional) : hexadecimal code of the notification's top line color, default corresponds to black
    """
    response = requests.post(
        url=webhook_url,
        headers={"Content-Type": "application/json"},
        json={
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "facts":[{
                    "name": "Project name",
                    "value": "Customer Churn"
                },{
                    "name": "Challenger_model_url",
                    "value": challenger_url
                },{
                    "name":"Message",
                    "value": content
                },{
                    "name":"Action",
                    "value": action
                }]
                
            }]
            
        }
    )
    return response.status_code # Should be 200



def model_training_reminder(webhook_url:str , content:str ,title:str, action :str, color:str="000000") -> int:
    """
      - Send a teams notification to the desired webhook_url
      - Returns the status code of the HTTP request
        - webhook_url : the url you got from the teams webhook configuration
        - content : your formatted notification content
        - title : the message that'll be displayed as title, and on phone notifications
        - action : action message
        - color (optional) : hexadecimal code of the notification's top line color, default corresponds to black
    """
    response = requests.post(
        url=webhook_url,
        headers={"Content-Type": "application/json"},
        json={
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "facts":[{
                    "name": "Project name",
                    "value": "Customer Churn"
                },{
                    "name":"Message",
                    "value": content
                },{
                    "name":"Action",
                    "value": action
                }]
                
            }]
            
        }
    )
    return response.status_code # Should be 200

def notify_new_region(webhook_url:str, content:str, title:str, action:str, color:str="000000") -> int:
    """
      - Send a teams notification to the desired webhook_url
      - Returns the status code of the HTTP request
      - webhook_url : the url you got from the teams webhook configuration
      - content : your formatted notification content
      - title : the message that'll be displayed as title, and on phone notifications
      - action : action message
      - color (optional) : hexadecimal code of the notification's top line color, default corresponds to black
    """
    response = requests.post(
        url=webhook_url,
        headers={"Content-Type": "application/json"},
        json={
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "facts":[{
                    "name": "Project name",
                    "value": "Customer Churn"
                },{
                    "name":"Message",
                    "value": content
                
                },{
                    "name":"Action",
                    "value": action
                
                }]
                
            }]
            
        }
    )
    return response.status_code # Should be 200
  
def pipeline_completion_notification(webhook_url:str , content:str ,title:str, action :str, color:str="000000") -> int:
    """
      - Send a teams notification to the desired webhook_url
      - Returns the status code of the HTTP request
        - webhook_url : the url you got from the teams webhook configuration
        - content : your formatted notification content
        - title : the message that'll be displayed as title, and on phone notifications
        - action : action message
        - color (optional) : hexadecimal code of the notification's top line color, default corresponds to black
    """
    response = requests.post(
        url=webhook_url,
        headers={"Content-Type": "application/json"},
        json={
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "facts":[{
                    "name": "Project name",
                    "value": "Customer Churn"
                },{
                    "name":"Message",
                    "value": content
                },{
                    "name":"Action",
                    "value": action
                }]
                
            }]
            
        }
    )
    return response.status_code # Should be 200