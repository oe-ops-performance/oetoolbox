from email import encoders
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path
from smtplib import SMTP

from .helpers import quiet_print_function

# note: computer must have necessary permissions for sending messages via SMTP
CORP_EXCH1_HOST = "smtp.corp.southwestgen.com"  # CORP-EXCH1.corp.SouthwestGen.com


# function for creating email file attachment "payload" object
def create_payload(filepath, filename=None):
    """Creates message payload object for file attachment

    Parameters
    ----------
    filepath : str | pathlib.Path
        The full path of the file for attachment. Supports CSV and Excel file types.
    filename : str, optional
        The name of the file attached to the email. Defaults to name from filepath.

    Returns
    -------
    MIMEApplication or MIMEBase object for use with MIMEMultipart
    """
    fpath = Path(filepath)
    if not any(x in fpath.suffix for x in ["xls", "csv", "txt"]):
        raise ValueError("Attachment must be either CSV or Excel format.")

    filename = fpath.name if filename is None else filename

    with open(filepath, "rb") as attachment:
        contents = attachment.read()

    if "xls" in fpath.suffix:
        payload = MIMEBase("application", "vnd.ms-excel")
        payload.add_header("content-disposition", "attachment", filename=filename)
        payload.set_payload(contents)
        encoders.encode_base64(payload)
    else:
        payload = MIMEApplication(contents, name=filename)

    return payload


# function for creating and sending an email with optional attachment(s)
def send_message(
    subject: str,
    body: str,
    addresses: dict,
    attachments: dict,
    host: str = None,
    q: bool = True,
    q2: bool = True,
):
    """Creates MIME message with subject and to/from/cc addresses (+ optional attachments)

    Parameters
    ----------
    subject : str
        The subject text for the email.
    body : str
        The body text for the email.
    addresses : dict
        Dictionary with keys=["from", "to", "cc"] and values=lists of email addresses.
    attachments : dict
        Dictionary with keys=filepaths (str) and values=filename (or None) for attachment.
    host : str, optional
        The SMTP mail server host name, defaults to None (uses CORP_EXCH1_HOST)
    q : bool, optional
        Quiet parameter; when False, enables status printouts. Defaults to True.
    q2 : bool, optional
        Quiet parameter 2; when False, enables SMTP status printouts. Defaults to True.


    Returns
    -------
    email.mime.multipart.MIMEMultipart object
    """
    qprint = lambda msg, end="\n": None if q else print(msg, end=end)

    # create message
    qprint("Creating message object")
    msg = MIMEMultipart()
    msg["subject"] = subject
    for key, val in addresses.items():
        msg[key] = ", ".join(val) if isinstance(val, list) else val

    message_body = MIMEText(body, "plain")
    msg.attach(payload=message_body)

    for filepath, filename in attachments.items():
        qprint("Adding attachment")
        payload = create_payload(filepath=filepath, filename=filename)
        msg.attach(payload=payload)

    # send message
    qprint("Sending message via smtp server\n")
    smtp_host = CORP_EXCH1_HOST if host is None else host
    try:
        with SMTP(host=smtp_host) as smtp:
            if not q2:
                smtp.set_debuglevel(1)
            smtp.send_message(msg=msg)
        qprint("\nMessage sent.")
    except Exception as e:
        qprint(e)

    return
