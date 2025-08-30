# security/auth.py

import hashlib
import hmac
import secrets
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

logger = logging.getLogger(__name__)

class SecurityManager:
    """Enhanced security manager for API authentication and authorization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('security', {})
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.token_expiry_hours = self.config.get('token_expiry_hours', 24)
        self.rate_limit_requests = self.config.get('rate_limit_requests', 100)
        self.rate_limit_window = self.config.get('rate_limit_window', 3600)  # 1 hour
        self.request_counts: Dict[str, Dict[str, Any]] = {}
        
    def generate_api_key(self) -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        return hmac.compare_digest(
            hashlib.sha256(provided_key.encode()).hexdigest(),
            stored_hash
        )
    
    def generate_jwt_token(self, user_id: str, permissions: list = None) -> str:
        """Generate JWT token for authenticated users."""
        payload = {
            'user_id': user_id,
            'permissions': permissions or [],
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_jwt_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = {
                'count': 1,
                'window_start': current_time
            }
            return True
        
        client_data = self.request_counts[client_ip]
        
        # Reset window if expired
        if current_time - client_data['window_start'] > self.rate_limit_window:
            client_data['count'] = 1
            client_data['window_start'] = current_time
            return True
        
        # Check if limit exceeded
        if client_data['count'] >= self.rate_limit_requests:
            return False
        
        client_data['count'] += 1
        return True
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input to prevent injection attacks."""
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$']
        sanitized = input_data
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        return sanitized.strip()


class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication."""
    
    def __init__(self, security_manager: SecurityManager, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.security_manager = security_manager
    
    async def __call__(self, request: Request) -> Optional[Dict[str, Any]]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
            payload = self.security_manager.verify_jwt_token(credentials.credentials)
            return payload
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")


def get_current_user(payload: Dict[str, Any] = Depends(JWTBearer)) -> Dict[str, Any]:
    """Dependency to get current authenticated user."""
    return payload
