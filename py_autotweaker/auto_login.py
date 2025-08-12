import os
from typing import Optional
import warnings
from save_design import fc_login

def auto_login_get_user_id() -> Optional[int]:
    """
    Return FC_USER_ID if it is set.
    Otherwise, attempt to automatically login using environment variables
    FC_USERNAME and FC_PASSWORD
    and return the user_id
    """
    user_id = os.environ.get('FC_USER_ID')
    if user_id:
        return user_id
    username = os.environ.get('FC_USERNAME')
    password = os.environ.get('FC_PASSWORD')
    if not username:
        warnings.warn('FC_USERNAME not set, cannot login. Alternatively, use FC_USER_ID')
        return None
    if not password:
        warnings.warn('FC_PASSWORD not set, cannot login. Alternatively, use FC_USER_ID')
        return None
    login_struct = fc_login(username, password)
    if not login_struct.success:
        warnings.warn('login failed')
    return login_struct.user_id