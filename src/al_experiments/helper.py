import logging
import requests

from al_experiments.config import DEBUG


def push_notification(text: str = "Execution finished.", is_error: bool = False) -> None:
    """ Helper method for push notifications.

    Args:
        text (str, optional): The text of the message. Defaults to "Execution finished.".
        is_error (bool, optional): Whether it is an error. Defaults to False.
    """

    if not is_error:
        logging.info(text)
    else:
        logging.error(text)

    if not DEBUG:
        with open("./secret", "r") as file:
            api_secret = str(file.read()).strip()
        headers = {"x-api-key": api_secret}
        message = {"title": "Jupyter Notebook", "body": text}
        requests.post(
            "https://push.techulus.com/api/v1/notify",
            data=message, headers=headers
        )
        del api_secret
