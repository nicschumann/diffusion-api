import secrets
from dotenv import dotenv_values
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

auth = HTTPBasic()
config = dotenv_values('.env.secret')

# security middleware
def check_credentials(credentials: HTTPBasicCredentials = Depends(auth)):

    # go ahead and factor these out...
    # un_correct = secrets.compare_digest(credentials.username, config['USERNAME'])
    pw_correct = secrets.compare_digest(credentials.password, config['PASSWORD'])

    if not pw_correct:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={'WWW-Authenticate': 'Basic'}
        )

    return credentials
