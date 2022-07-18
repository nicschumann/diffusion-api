import secrets
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

auth = HTTPBasic()

# security middleware
def check_credentials(credentials: HTTPBasicCredentials = Depends(auth)):

    # go ahead and factor these out...
    un_correct = secrets.compare_digest(credentials.username, 'nic')
    pw_correct = secrets.compare_digest(credentials.password, 'test')

    if not (un_correct and pw_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={'WWW-Authenticate': 'Basic'}
        )

    return credentials
