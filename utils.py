import random
from datetime import datetime
import logging
import os
from email.message import EmailMessage

logger = logging.getLogger("devset")


def generate_otp(length: int = 6) -> str:
    """Generate a numeric OTP of given length."""
    range_start = 10 ** (length - 1)
    range_end = (10 ** length) - 1
    return str(random.randint(range_start, range_end))


async def send_email(to_email: str, subject: str, body: str) -> None:
    """Send email using SMTP asynchronously via aiosmtplib.

    Uses environment variables: SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM.
    If SMTP_HOST is not configured, falls back to logging the message.
    """
    smtp_host = os.getenv("SMTP_HOST")
    if not smtp_host:
        logger.info("[send_email - simulated] To: %s Subject: %s Body: %s", to_email, subject, body)
        return

    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM", smtp_user or f"no-reply@{smtp_host}")

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        # import aiosmtplib lazily so running tests without installing SMTP deps still works
        import aiosmtplib

        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_user,
            password=smtp_pass,
            start_tls=True,
        )
        logger.info("[send_email] Sent to %s via %s:%s", to_email, smtp_host, smtp_port)
    except Exception as e:
        logger.exception("Failed to send email to %s: %s", to_email, e)


def now_utc() -> datetime:
    return datetime.utcnow()
