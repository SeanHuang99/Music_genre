import requests

def send_pushbullet_notification(title, body):
    access_token = 'o.2JmgNC14IhIf3Odg5nK735J7L1SagkmR'  # 替换为你从 Pushbullet 获取的 Access Token
    data = {
        "type": "note",
        "title": title,
        "body": body
    }
    headers = {
        "Access-Token": access_token,
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.pushbullet.com/v2/pushes", json=data, headers=headers)
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
    else:
        print("Notification sent successfully")

# # 调用函数发送通知
# send_pushbullet_notification("Task completed", "Your task on the server has finished.")
