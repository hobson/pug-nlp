import random
import string

def fedora_password_salt(length=8, alphabet=string.letters + string.digits + './'):
    """Generate a random salt string for use in `crypt.crypt(password, salt)`"""
    return ''.join(random.choice(alphabet) for position in range(length))

