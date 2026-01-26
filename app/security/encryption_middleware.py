from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response
from app.core.settings import settings
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
import json
import binascii
import base64

class EncryptionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.key = binascii.unhexlify(settings.ENCRYPTION_KEY_HEX)

    def encrypt(self, data: str) -> str:
        iv = os.urandom(16)
        
        # Logica compatível com decryption frontend (AES-CBC + PKCS7)
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data.encode()) + padder.finalize()

        # Encrypt
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return Base64(IV + Ciphertext)
        return base64.b64encode(iv + ciphertext).decode('utf-8')

    async def dispatch(self, request: Request, call_next):
        # Ignorar rotas de documentação e schema
        if request.url.path in ["/openapi.json", "/docs", "/redoc"]:
            return await call_next(request)

        response: Response = await call_next(request)
        
        # Only encrypt JSON responses
        if response.headers.get("content-type") == "application/json":
            # Consuming the response body
            body = [section async for section in response.body_iterator]
            body_content = b"".join(body).decode()
            
            try:
                # Ensure valid JSON before encrypting
                json.loads(body_content)
                encrypted_content = self.encrypt(body_content)
                
                new_headers = dict(response.headers)
                if "content-length" in new_headers:
                    del new_headers["content-length"]

                return Response(
                    content=json.dumps({"encrypted_data": encrypted_content}),
                    headers=new_headers,
                    status_code=response.status_code,
                    media_type="application/json"
                )
            except json.JSONDecodeError:
                # If it's not valid JSON, return original response (reconstructed)
                return Response(
                    content=body_content,
                    headers=dict(response.headers),
                    status_code=response.status_code,
                    media_type="application/json"
                )
                
        return response
